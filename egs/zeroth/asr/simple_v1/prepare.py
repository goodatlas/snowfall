from collections import defaultdict

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet

from pathlib import Path
from lhotse.kaldi import load_kaldi_data_dir
from lhotse.recipes.utils import read_manifests_if_cached
from lhotse.utils import Pathlike
from typing import Dict, Optional, Sequence, Tuple, Union
from lhotse.recipes import prepare_musan
from lhotse import CutSet, Fbank, FbankConfig, LilcomHdf5Writer, combine
from concurrent.futures.thread import ThreadPoolExecutor
from tqdm.auto import tqdm

from contextlib import contextmanager

import os
import sys
import torch
import subprocess
import argparse
import logging
import re
import morfessor

from morfessor import BaselineModel

# Torch's multithreaded behavior needs to be disabled or it wastes a lot of CPU and
# slow things down.  Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@contextmanager
def get_executor():
    # We'll either return a process pool or a distributed worker pool.
    # Note that this has to be a context manager because we might use multiple
    # context manager ("with" clauses) inside, and this way everything will
    # free up the resources at the right time.
    try:
        # If this is executed on the CLSP grid, we will try to use the
        # Grid Engine to distribute the tasks.
        # Other clusters can also benefit from that, provided a cluster-specific wrapper.
        # (see https://github.com/pzelasko/plz for reference)
        #
        # The following must be installed:
        # $ pip install dask distributed
        # $ pip install git+https://github.com/pzelasko/plz

        # name = subprocess.check_output('hostname -f', shell=True, text=True)
        # if name.strip().endswith('jupiter'):
        import plz
        from distributed import Client
        with plz.setup_cluster() as cluster:
            cluster.scale(80)
            yield Client(cluster)
        return
    except:
        pass
    # No need to return anything - compute_and_store_features
    # will just instantiate the pool itself.
    yield None


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--num-jobs',
        type=int,
        default=min(15, os.cpu_count()),
        help='number of jobs for feature extraction')
    parser.add_argument(
        '--morpheme-analysis-model-path',
        type=str,
        default='data/local/lm/zeroth_morfessor.seg'
    )
    parser.add_argument(
        '--word-boundary-symbol',
        type=str,
        default='',
        help="if you want to leave word boundary symbol after morpheme segmentation, set it '|'"
    )
    return parser


def import_kaldi_data_dir(
    data_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]],
    manifest_dir: Pathlike,
    sampling_rate: int = 16000
) -> None:

    manifests = defaultdict(dict)
    for part in dataset_parts:
        _dir = data_dir + "/" + part
        print(f'started import kaldi data dir in {_dir}')

        recording_set, maybe_supervision_set, maybe_feature_set = load_kaldi_data_dir(
            path=_dir,
            sampling_rate=sampling_rate
        )
        manifest_dir = Path(manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        recording_set.to_file(manifest_dir / f'recordings_{part}.json')

        if maybe_supervision_set is not None:
            maybe_supervision_set.to_file(
                manifest_dir / f'supervisions_{part}.json')

        manifests[part] = {
            'recordings': recording_set,
            'supervisions': maybe_supervision_set
        }

    return dict(manifests)


def prepare_zeroth(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]],
    output_dir: Pathlike,
    morpheme_analysis_model_path: Optional[Pathlike],
    word_boundary_symbol: Optional[str],
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:

    corpus_dir = Path(corpus_dir)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        maybe_manifests = read_manifests_if_cached(dataset_parts=dataset_parts,
                                                   output_dir=output_dir)
        if maybe_manifests is not None:
            print("read_manifests_if_cached() works")
            return maybe_manifests

    speaker_info_path = corpus_dir / "SPEAKERS"
    if os.path.exists(speaker_info_path):
        speakers = {}
        print(f'found speaker_info file: {speaker_info_path}')
        with open(speaker_info_path) as f:
            for line in f:
                splits = line.strip().split('|')
                speaker_id = splits[0]
                speaker_name = splits[1]
                gender = splits[2]
                script_id = splits[3]
                speakers.setdefault(speaker_id, {})
                speakers[speaker_id]["name"] = speaker_name
                speakers[speaker_id]["gender"] = gender
                speakers[speaker_id]["script_id"] = script_id

    if os.path.exists(morpheme_analysis_model_path):
        # load model
        print(
            f'Loading morpheme analysis model: {morpheme_analysis_model_path}')
        io = morfessor.MorfessorIO()
        model = io.read_binary_model_file(morpheme_analysis_model_path)

        print(f'word boundary symbol: {word_boundary_symbol}')

    manifests = defaultdict(dict)
    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc='Dataset parts'):
            recordings = []
            supervisions = []
            part_path = corpus_dir / part  # corpus/recData01
            futures = []
            for trans_path in tqdm(part_path.rglob('*.txt'), desc='Distributing tasks', leave=False):
                # corpus/recData01/001/014
                #   014_001.trans.txt
                #   014_001_001.flac
                with open(trans_path) as f:
                    for line in f:
                        futures.append(
                            ex.submit(parse_utterance, part_path, line, speakers, model, word_boundary_symbol))

            for future in tqdm(futures, desc='Processing', leave=False):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                recordings.append(recording)
                supervisions.append(segment)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

            validate_recordings_and_supervisions(
                recording_set, supervision_set)

            if output_dir is not None:
                supervision_set.to_json(
                    output_dir / f'supervisions_{part}.json')
                recording_set.to_json(output_dir / f'recordings_{part}.json')

            manifests[part] = {
                'recordings': recording_set,
                'supervisions': supervision_set
            }

    return dict(manifests)  # Convert to normal dict

    # return import_kaldi_data_dir(data_dir, dataset_parts, output_dir)


def parse_utterance(
        dataset_split_path: Path,
        line: str,
        speakers: Optional[Dict],
        model: Optional[BaselineModel],
        word_boundary_symbol: Optional[str]
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    recording_id, text = line.strip().split(maxsplit=1)
    # recording_id, text
    # 014_001_001 네 합정동이요 네 잠시만요 네 말씀해주세요 다진마늘 국산 하나 무 한포 세제 이 키로짜리 두 개 쇠고기 다시다
    # 014_001_002 저희는 상담사가요 열 명 정도 됩니다 사장님 배추 한 포 배추 아직까지 망으로 나와요 네 아직까지 망으로 나와요

    if model is not None:
        smooth = 0
        maxlen = 30
        analyzed_text = ""
        for word in text.split():
            constructions, logp = model.viterbi_segment(
                word, smooth, maxlen)
            analyzed_text += word_boundary_symbol + \
                " ".join(constructions) + " "
        text = analyzed_text.strip()
        custom = "morpheme_updated"

    # Create the Recording first
    speaker_id = recording_id.split('_')[0]
    script_id = recording_id.split('_')[1]
    utt_id = recording_id.split('_')[2]

    audio_path = dataset_split_path / \
        Path(script_id + "/" + speaker_id + "/" +
             utt_id).parent / f'{recording_id}.flac'
    if not audio_path.is_file():
        logging.warning(f'No such file: {audio_path}')
        return None
    recording = Recording.from_file(audio_path, recording_id=recording_id)

    # Then, create the corresponding supervisions
    if speakers is not None:
        speaker = speakers[speaker_id]["name"]
        gender = speakers[speaker_id]["gender"]

    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language='Korean',
        speaker=re.sub(
            r'-.*', r'', recording.id) if speaker is None else speaker,
        gender=None if gender is None else gender,
        custom=None if custom is None else custom,
        text=text.strip()
    )
    return recording, segment


def locate_corpus(*corpus_dirs):
    for d in corpus_dirs:
        if os.path.exists(d):
            return d
    print("Please create a place on your system to put the downloaded Librispeech data "
          "and add it to `corpus_dirs`")
    sys.exit(1)


# def update_segmentation(
#    model_path: Pathlike,
#    manifests: Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]
# ):
#
#    # load model
#    print(f'Loading morpheme analysis model: {model_path}')
#    io = morfessor.MorfessorIO()
#    model = io.read_binary_model_file(model_path)
#
#    smooth = 0
#    maxlen = 30
#    word_boundary_symbol = "|"
#
#    for part in manifests.keys():
#        print(f"Started processing: {part}")
#        for segment in manifests[part]['supervisions']:
#            if segment.custom is None:
#                analyzed_text = ""
#                for word in segment.text.split():
#                    constructions, logp = model.viterbi_segment(
#                        word, smooth, maxlen)
#                    analyzed_text += word_boundary_symbol + \
#                        " ".join(constructions) + " "
#                analyzed_text = analyzed_text.strip()
#                manifests[part]['supervisions'].text = analyzed_text
#                manifests[part]['supervisions'].custom = "morpheme_updated"


def main():
    args = get_parser().parse_args()

    musan_dir = locate_corpus(
        Path('/mnt/data/reference/musan'),
    )
    output_dir = Path('exp/data')

    print('Zeroth manifest preparation:')
    zeroth_manifests = prepare_zeroth(
        'corpus',
        ('recData01', 'recData02', 'recData03', 'testData01', 'testData02'),
        output_dir,
        args.morpheme_analysis_model_path,
        args.word_boundary_symbol)

    # morpheme analysis
    # update_segmentation(
    #    'data/local/lm/zeroth_morfessor.seg',
    #    zeroth_manifests
    # )

    print('Musan manifest preparation:')
    musan_cuts_path = output_dir / 'cuts_musan.json.gz'
    musan_manifests = prepare_musan(
        corpus_dir=musan_dir,
        output_dir=output_dir,
        parts=('music', 'speech', 'noise')
    )

    print('Feature extraction:')
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    with get_executor() as ex:  # Initialize the executor only once.
        for partition, manifests in zeroth_manifests.items():
            if (output_dir / f'cuts_{partition}.json.gz').is_file():
                print(f'{partition} already exists - skipping.')
                continue
            print('Processing', partition)
            cut_set = CutSet.from_manifests(
                recordings=manifests['recordings'],
                supervisions=manifests['supervisions']
            )
            if 'train' in partition:
                cut_set = cut_set + \
                    cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f'{output_dir}/feats_{partition}',
                # when an executor is specified, make more partitions
                num_jobs=args.num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomHdf5Writer
            )
            zeroth_manifests[partition]['cuts'] = cut_set
            cut_set.to_json(output_dir / f'cuts_{partition}.json.gz')
        # Now onto Musan
        if not musan_cuts_path.is_file():
            print('Extracting features for Musan')
            # create chunks of Musan with duration 5 - 10 seconds
            musan_cuts = CutSet.from_manifests(
                recordings=combine(part['recordings']
                                   for part in musan_manifests.values())
            ).cut_into_windows(10.0).filter(lambda c: c.duration > 5).compute_and_store_features(
                extractor=extractor,
                storage_path=f'{output_dir}/feats_musan',
                num_jobs=args.num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomHdf5Writer
            )
            musan_cuts.to_json(musan_cuts_path)


if __name__ == '__main__':
    main()
