head_list = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def trans_f0_seq(feature_pit, transform):
    feature_pit = feature_pit * 2 ** (transform / 12)
    return round(feature_pit, 1)


def move_key(raw_data, mv_key):
    head = raw_data[:-1]
    body = int(raw_data[-1])
    new_head_index = head_list.index(head) + mv_key
    while new_head_index < 0:
        body -= 1
        new_head_index += 12
    while new_head_index > 11:
        body += 1
        new_head_index -= 12
    result_data = head_list[new_head_index] + str(body)
    return result_data


def trans_key(raw_data, key):
    for i in raw_data:
        note_seq_list = i["note_seq"].split(" ")
        new_note_seq_list = []
        for note_seq in note_seq_list:
            if note_seq != "rest":
                new_note_seq = move_key(note_seq, key)
                new_note_seq_list.append(new_note_seq)
            else:
                new_note_seq_list.append(note_seq)
        i["note_seq"] = " ".join(new_note_seq_list)

        f0_seq_list = i["f0_seq"].split(" ")
        f0_seq_list = [float(x) for x in f0_seq_list]
        new_f0_seq_list = []
        for f0_seq in f0_seq_list:
            new_f0_seq = trans_f0_seq(f0_seq, key)
            new_f0_seq_list.append(str(new_f0_seq))
        i["f0_seq"] = " ".join(new_f0_seq_list)
    return raw_data


key = -6
f_w = open("raw.txt", "w", encoding='utf-8')
with open("result.txt", "r", encoding='utf-8') as f:
    raw_data = f.readlines()
    for raw in raw_data:
        raw_list = raw.split("|")
        new_note_seq_list = []
        for note_seq in raw_list[3].split(" "):
            if note_seq != "rest":
                note_seq = note_seq.split("/")[0] if "/" in note_seq else note_seq
                new_note_seq = move_key(note_seq, key)
                new_note_seq_list.append(new_note_seq)
            else:
                new_note_seq_list.append(note_seq)
        raw_list[3] = " ".join(new_note_seq_list)
        f_w.write("|".join(raw_list))
f_w.close()
