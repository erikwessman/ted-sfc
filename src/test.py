from typing import List, Tuple


class Sequence:
    def __init__(self, path, interrupts=0):
        self.path = path
        self.interrupts = interrupts
        self.active = True


def frames_to_seconds(nr_frames: int, fps: int):
    return nr_frames / fps


def get_ordered_sequences(path: List[Tuple[int, int]], tolerance=1):
    active_sequences: List[Sequence] = []
    ordered_sequences: List[Sequence] = []

    for node in path:

        has_created_new_sequence = False

        if not active_sequences:
            seq = Sequence([node])
            active_sequences.append(seq)
            has_created_new_sequence = True
            continue

        active_sequences_copy = active_sequences.copy()

        for i, active_seq in enumerate(active_sequences_copy):
            if not active_seq.active:
                continue

            prev = active_seq.path[-1]

            if node[0] > prev[0]:
                active_seq.path.append(node)
            elif node[0] == prev[0]:
                frame_diff = abs(node[1] - prev[1])

                if frames_to_seconds(frame_diff, 10) > 3:
                    active_seq.active = False
                    ordered_sequences.append(active_seq)
            else:
                active_seq.interrupts += 1

                if active_seq.interrupts > tolerance:
                    active_seq.active = False
                    ordered_sequences.append(active_seq)

                if has_created_new_sequence:
                    continue

                seq = Sequence([node])
                active_sequences.append(seq)
                has_created_new_sequence = True

    for active_seq in active_sequences:
        if len(active_seq.path) > 1:
            ordered_sequences.append(active_seq)
            active_seq.active = False

    return ordered_sequences


data = [(1, 2), (2, 3), (3, 4), (2, 5), (1, 6), (5, 7), (1, 8), (2, 9)]
p = get_ordered_sequences(data)


for o in p:
    print(o.path)
