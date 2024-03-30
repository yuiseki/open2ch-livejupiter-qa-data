import datasets

livejupiter_original_data = datasets.load_dataset(
    "p1atdev/open2ch", "livejupiter", split="train")
print(livejupiter_original_data)

livejupiter_qa_pair = []
for data in livejupiter_original_data:
    contents = data['dialogue']["content"]
    if len(contents) != 2:
        continue
    # contents[0]に?や？が含まれていない場合はスキップ
    if contents[0].find("?") == -1 and contents[0].find("？") == -1:
        continue
    # contents[0]に改行が含まれている場合はスキップ
    if contents[0].find("\n") != -1:
        continue
    # contentsが" \n "になっているので"\n"に変換
    livejupiter_qa_pair.append({
        "question": contents[0].replace(" \n ", "\n").replace("\n ", "\n"),
        "answer": contents[1].replace(" \n ", "\n").replace("\n ", "\n")
    })

print(len(livejupiter_qa_pair))


livejupiter_qa_dataset = datasets.Dataset.from_list(livejupiter_qa_pair)
print(livejupiter_qa_dataset)

livejupiter_qa_dataset.push_to_hub("yuiseki/open2ch-livejupiter-qa")
