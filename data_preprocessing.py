import os
import time
import glob
import json
import random
import regex as re
import pandas as pd
from tqdm import tqdm

def divide_source_file_list(l, n): 
    
  for i in range(0, len(l), n): 
   yield l[i:i + n] 

def sorted_list(path_list):
    
    path_list = sorted(path_list, reverse=False)
    path_list = sorted(path_list, key=len)
    
    return path_list


def json_file_path_list(path_list):
    
    file_path  = [glob.glob(i, recursive = True) for i in path_list][0]
    file_path = sorted_list(file_path)
    
    return file_path


def train_valid_json_file_path_list(path_list):

    train_file_path, valid_file_path = [glob.glob(i, recursive = True) if 'rain' in i
                                        else glob.glob(i, recursive = True)
                                        for i in path_list]

    train_part1_file_path = [train_file for train_file in train_file_path if 'Part2' not in train_file]  #
    valid_part1_file_path = [valid_file for valid_file in valid_file_path if 'Part2' not in valid_file]   #  

    train_part2_file_path = [train_file for train_file in train_file_path if 'Part2' in train_file]  #
    valid_part2_file_path = [valid_file for valid_file in valid_file_path if 'Part2' in valid_file]   #  

    # train_file_path = [train_file for train_file in train_file_path if '_Clickbait_Auto' not in train_file]
    # valid_file_path = [valid_file for valid_file in valid_file_path if '_Clickbait_Auto' not in valid_file]
    # train_file_path = [train_file for train_file in train_file_path if 'Part2' not in train_file]  #
    # valid_file_path = [valid_file for valid_file in valid_file_path if 'Part2' not in valid_file]   #  
    # train_file_path = sorted_list(train_file_path)
    # valid_file_path = sorted_list(valid_file_path)
   
    train_part1_file_path = sorted_list(train_part1_file_path)
    valid_part1_file_path = sorted_list(valid_part1_file_path)
    train_part2_file_path = sorted_list(train_part2_file_path)
    valid_part2_file_path = sorted_list(valid_part2_file_path)

    return train_part1_file_path, valid_part1_file_path, train_part2_file_path, valid_part2_file_path


def xlsx_file_path_list(file_path, folder_corpus_type_path):
    
    xlsx_file_path = [folder_corpus_type_path + str(i) + ".xlsx"
                                for i in range((len(file_path) // 1000) + 1 )]
        
    return xlsx_file_path


def divide_clickbait_or_nonclibait_json_file_path_list(criterion, json_file_list):

    clickbait_path, nonclickbait_path = [], []
    for x in json_file_list:
        (clickbait_path, nonclickbait_path)[criterion in x].append(x)

    return clickbait_path, nonclickbait_path

def make_train_valid_json_xlsx_file_path_list(json_path_list, xlsx_path_list):

    train_part1_json_file_path, valid_part1_json_file_path, \
    train_part2_json_file_path, valid_part2_json_file_path = train_valid_json_file_path_list(json_path_list)

    train_part1_clickbait_json_file_path, train_part1_nonclickbait_json_file_path = divide_clickbait_or_nonclibait_json_file_path_list("Non", train_part1_json_file_path)
    valid_part1_clickbait_json_file_path, valid_part1_nonclickbait_json_file_path = divide_clickbait_or_nonclibait_json_file_path_list("Non", valid_part1_json_file_path)

    train_part2_clickbait_json_file_path, train_part2_nonclickbait_json_file_path = divide_clickbait_or_nonclibait_json_file_path_list("Non", train_part2_json_file_path)
    valid_part2_clickbait_json_file_path, valid_part2_nonclickbait_json_file_path = divide_clickbait_or_nonclibait_json_file_path_list("Non", valid_part2_json_file_path)

    the_number_of_train_clickbait_json_file = len(train_part1_clickbait_json_file_path) + len(train_part2_clickbait_json_file_path)
    the_number_of_valid_clickbait_json_file = len(valid_part1_clickbait_json_file_path) + len(valid_part2_clickbait_json_file_path)
    the_number_of_clickbait_json_file = the_number_of_train_clickbait_json_file + the_number_of_valid_clickbait_json_file
    print("The number of train clickbait json file:", the_number_of_train_clickbait_json_file)
    print("The number of valid clickbait json file:", the_number_of_valid_clickbait_json_file)
    print("The number of clickbait json file:", the_number_of_clickbait_json_file)

    the_number_of_train_nonclickbait_json_file  = len(train_part1_nonclickbait_json_file_path) + len(train_part2_nonclickbait_json_file_path)
    the_number_of_valid_nonclickbait_json_file  = len(valid_part1_nonclickbait_json_file_path) + len(valid_part2_nonclickbait_json_file_path)
    the_number_of_nonclickbait_json_file = the_number_of_train_nonclickbait_json_file + the_number_of_valid_nonclickbait_json_file
    print()
    print("The number of train nonclickbait json file:", the_number_of_train_nonclickbait_json_file)
    print("The number of valid nonclickbait json file:", the_number_of_valid_nonclickbait_json_file)
    print("The number of nonclickbait json file:", the_number_of_nonclickbait_json_file)

    the_number_of_train_json_file  = len(train_part1_json_file_path) + len(train_part2_json_file_path)
    the_number_of_valid_json_file  = len(valid_part1_json_file_path) + len(valid_part2_json_file_path)
    the_number_of_json_file = the_number_of_train_json_file + the_number_of_valid_json_file
    print()
    print("The number of train json file:", the_number_of_train_json_file)
    print("The number of valid json file:", the_number_of_valid_json_file)
    print("The number of json file:", the_number_of_json_file)
    
    train_part1_clickbait_xlsx_file_path = xlsx_file_path_list(train_part1_clickbait_json_file_path, xlsx_path_list[0])
    valid_part1_clickbait_xlsx_file_path = xlsx_file_path_list(valid_part1_clickbait_json_file_path, xlsx_path_list[1])
    train_part2_clickbait_xlsx_file_path = xlsx_file_path_list(train_part2_clickbait_json_file_path, xlsx_path_list[2])
    valid_part2_clickbait_xlsx_file_path = xlsx_file_path_list(valid_part2_clickbait_json_file_path, xlsx_path_list[3])
   
    train_part1_nonclickbait_xlsx_file_path = xlsx_file_path_list(train_part1_nonclickbait_json_file_path, xlsx_path_list[4])
    valid_part1_nonclickbait_xlsx_file_path = xlsx_file_path_list(valid_part1_nonclickbait_json_file_path, xlsx_path_list[5])
    train_part2_nonclickbait_xlsx_file_path = xlsx_file_path_list(train_part2_nonclickbait_json_file_path, xlsx_path_list[6])
    valid_part2_nonclickbait_xlsx_file_path = xlsx_file_path_list(valid_part2_nonclickbait_json_file_path, xlsx_path_list[7])


    return train_part1_clickbait_json_file_path, train_part1_nonclickbait_json_file_path, \
    train_part2_clickbait_json_file_path, train_part2_nonclickbait_json_file_path, \
    valid_part1_clickbait_json_file_path, valid_part1_nonclickbait_json_file_path, \
    valid_part2_clickbait_json_file_path, valid_part2_nonclickbait_json_file_path, \
    train_part1_clickbait_xlsx_file_path, train_part1_nonclickbait_xlsx_file_path, \
    train_part2_clickbait_xlsx_file_path, train_part2_nonclickbait_xlsx_file_path, \
    valid_part1_clickbait_xlsx_file_path, valid_part1_nonclickbait_xlsx_file_path, \
    valid_part2_clickbait_xlsx_file_path, valid_part2_nonclickbait_xlsx_file_path, \
    

def preprocess_source(source):
    
    source = re.sub(r"\[.*?\]|\{.*?\}|\(.*?\)", "", source)
    source = source.replace("\n", " ")
    source = source.replace("\\\\", "")
    source = source.replace('"', "")
    source = source.replace("'", "")
    source = re.sub(r"[^A-Za-z0-9ㄱ-ㅎ가-힣一-鿕㐀-䶵豈-龎()+-:.,]", " ", source)

    return source

def make_source_special01(json_sample):
    try:
        title = json_sample['labeledDataInfo']['newTitle']
    except:
        title = json_sample['sourceDataInfo']['newsTitle']
    content = json_sample['sourceDataInfo']['newsContent']
    # label = json_sample['sourceDataInfo']['newsCategory']
    
    # title = preprocess_source(title)
    # content = preprocess_source(content)

    # source = "[CLS]" + " " + title + " " + "[SEP]" + " " + content + " " + "[SEP]"
    
    return title, content

def make_source_special03(json_sample):
    try:
        title = json_sample['labeledDataInfo']['newTitle']
    except:
        title = json_sample['sourceDataInfo']['newsTitle']
    contents = json_sample['sourceDataInfo']["sentenceInfo"]
    # label = json_sample['sourceDataInfo']['newsCategory']
    text = ""
    # num = 0
    for content in contents:
        # num += 1
        # if num == 1:
        text += preprocess_source(content["sentenceContent"]) + " "
        # elif num != 1:
        #     pass
    
    if (title[0] == "[" and title[-1] == "]") or (title[0] == "{" and title[-1] == "}") or (title[0] == "(" and title[-1] == ")"):
        pass
    else:
        title = preprocess_source(title)


    # contents = preprocess_source(contents)

    # total_contents +=  "\n" + preprocess_source(content["sentenceContent"])

    # source = "[CLS]" + " " + title + " " + "[SEP]" + " " + lead + " " + "[SEP]"

    return title, text

def make_source_special02(json_sample):
    try:
        title = json_sample['labeledDataInfo']['newTitle']
    except:
        title = json_sample['sourceDataInfo']['newsTitle']
    contents = json_sample['sourceDataInfo']["sentenceInfo"]
    # label = json_sample['sourceDataInfo']['newsCategory']

    lead2 = ""
    lead3 = ""
    num = 0
    for content in contents:
        num += 1
        if num == 1:
            lead1 = content["sentenceContent"]
            lead2 += content["sentenceContent"] + " "
            lead3 += content["sentenceContent"] + " "
        elif num == 2:
            lead2 += content["sentenceContent"] + " "
            lead3 += content["sentenceContent"] + " "
        elif num == 3:
            lead3 += content["sentenceContent"] + " "
        elif num >= 4:
            pass
    """
    if (title[0] == "[" and title[-1] == "]") or (title[0] == "{" and title[-1] == "}") or (title[0] == "(" and title[-1] == ")"):
        pass
    else:
        title = preprocess_source(title)
    """
 
    # total_contents +=  "\n" + preprocess_source(content["sentenceContent"])

    # source = "[CLS]" + " " + title + " " + "[SEP]" + " " + lead + " " + "[SEP]"

    return title, lead1, lead2, lead3



"""
def make_source_special02(json_sample):
    title = json_sample['sourceDataInfo']['newsTitle']
    real_contents = json_sample['sourceDataInfo']["sentenceInfo"]
    fake_contents = json_sample['labeledDataInfo']["processSentenceInfo"]
    label = json_sample['sourceDataInfo']['newsCategory']

    num = 0
    for content in real_contents:
        num += 1
        if num == 1:
            real_lead = preprocess_source(content["sentenceContent"])
        elif num != 1:
            pass

    try:
        fake_leads = []
        for content in fake_contents:
            if content["subjectConsistencyYn"] == "N":
                fake_leads.append(content["sentenceContent"])
        fake_lead = fake_leads[0]
    except:
        fake_lead = "not fake"

    title = preprocess_source(title)
    real_lead = preprocess_source(real_lead)
    fake_lead = preprocess_source(fake_lead)
            # total_contents +=  "\n" + preprocess_source(content["sentenceContent"])

    # source = "[CLS]" + " " + title + " " + "[SEP]" + " " + lead + " " + "[SEP]"

    return title, real_lead, fake_lead
"""
def write_jsontext_to_xlsx_file_with_batch_size(source_file_list, xlsx_file_path_list, batch_size):

    progress_length = len(source_file_list) // batch_size
    print("[Size]")
    print("The number of xlsx file: " + str(progress_length))
    print("\n[Order]")
    # source_list = []
    # content_list = []


    # source_special02_list = []
    # real_lead_special02_list = []
    # fake_lead_special02_list = []

    source_file_list = list(divide_source_file_list(source_file_list, batch_size))
    pbar = tqdm(range(len(source_file_list)))
    num = -1

    for i in pbar:
        num += 1

        source_files = source_file_list[num]

        source_special01_list = []
        lead_special01_list = []
        source_special02_list = []
        lead1_list = []
        lead2_list = []
        lead3_list = []
        label_list = []

        for idx, source_file in enumerate(source_files):
            
            with open(source_file, 'r', encoding='utf-8') as one_json_file:
                one_json_sample = json.load(one_json_file)
            
            # source, content, label = make_source(one_json_sample)
            if "Part1" in source_file:
                source_special01, lead_special01 = make_source_special01(one_json_sample)
                source_special02, lead1, lead2, lead3 = make_source_special02(one_json_sample)
                # source_list.append(source)
                source_special01_list.append(source_special01)
                source_special02_list.append(source_special02)
                # content_list.append(content)
                lead_special01_list.append(lead_special01)
                lead1_list.append(lead1)
                lead2_list.append(lead2)
                lead3_list.append(lead3)

                if 'Non' in source_file:
                    label_list.append("real")           
                elif 'Non' not in source_file:
                    label_list.append("fake") 

        # source_df = pd.DataFrame({'Title':source_list, 'Content':content_list, 'Label':label_list})
        # source_special01_df = pd.DataFrame({'Title':source_special01_list, 'Content':lead_special01_list, 'Fake Content':fake_content_list,  'Label':label_list})
        source_special01_df = pd.DataFrame({'Title':source_special01_list, 'Content':lead_special01_list, 'Label':label_list})
        source_special02_df = pd.DataFrame({'Title':source_special02_list, 'Content':lead1_list, 'Label':label_list})
        source_special03_df = pd.DataFrame({'Title':source_special02_list, 'Content':lead2_list, 'Label':label_list})
        source_special04_df = pd.DataFrame({'Title':source_special02_list, 'Content':lead3_list, 'Label':label_list})
        source_df_path = xlsx_file_path_list[num]
        
        # source_df.to_excel(source_df_path.replace(".xlsx", "_not_special.xlsx"), index=False)
        # source_df.to_csv(source_df_path.replace(".xlsx", "_not_special.csv"), index=False)
        source_special01_df.to_csv(source_df_path.replace(".xlsx", "_special_01.csv"), index=False)
        source_special02_df.to_csv(source_df_path.replace(".xlsx", "_special_02.csv"), index=False)
        source_special03_df.to_csv(source_df_path.replace(".xlsx", "_special_03.csv"), index=False)
        source_special04_df.to_csv(source_df_path.replace(".xlsx", "_special_04.csv"), index=False)
            
        """
            source_special01_list = []
            lead_special01_list = []
            source_special02_list = []
            lead1_list = []
            lead2_list = []
            lead3_list = []
            # lead_special02_list = []

        elif idx == (len(source_file_list) -1): 
            # source_list.append(source)
            source_special01_list.append(source_special01)
            source_special02_list.append(source_special02)
            # content_list.append(content)
            lead_special01_list.append(lead_special01)
            lead1_list.append(lead1)
            lead2_list.append(lead2)
            lead3_list.append(lead3)

            num += 1
            if 'Non' in source_file:
                # label_list = ["진짜 " + label] * len(source_list)
                label_list = ["real"] * len(source_special01_list)

            elif 'Non' not in source_file:
                # label_list = ["가짜 " + label]  * len(source_list)
                label_list = ["fake"] * len(source_special01_list)

            # fake_content_list = ["not fake"] * len(source_special01_list)

            # source_df = pd.DataFrame({'Title':source_list, 'Content':content_list, 'Label':label_list})
            # source_special01_df = pd.DataFrame({'Title':source_special01_list, 'Content':lead_special01_list, 'Fake Content':fake_content_list,  'Label':label_list})
            source_special01_df = pd.DataFrame({'Title':source_special01_list, 'Content':lead_special01_list, 'Label':label_list})
            source_special02_df = pd.DataFrame({'Title':source_special02_list, 'Content':lead1_list, 'Label':label_list})
            source_special03_df = pd.DataFrame({'Title':source_special02_list, 'Content':lead2_list, 'Label':label_list})
            source_special04_df = pd.DataFrame({'Title':source_special02_list, 'Content':lead3_list, 'Label':label_list})
            source_df_path = xlsx_file_path_list[num]
            
            # source_df.to_excel(source_df_path.replace(".xlsx", "_not_special.xlsx"), index=False)
            # source_df.to_csv(source_df_path.replace(".xlsx", "_not_special.csv"), index=False)
            source_special01_df.to_csv(source_df_path.replace(".xlsx", "_special_01.csv"), index=False)
            source_special02_df.to_csv(source_df_path.replace(".xlsx", "_special_02.csv"), index=False)
            source_special03_df.to_csv(source_df_path.replace(".xlsx", "_special_03.csv"), index=False)
            source_special04_df.to_csv(source_df_path.replace(".xlsx", "_special_04.csv"), index=False)

        # source_list.append(source)
        source_special01_list.append(source_special01)
        source_special02_list.append(source_special02)
        # content_list.append(content)
        lead_special01_list.append(lead_special01)
        lead1_list.append(lead1)
        lead2_list.append(lead2)
        lead3_list.append(lead3)

    # pbar.n += 1
    # pbar.refresh()
    # time.sleep(0.01)          

    
        elif "Part2" in source_file:
            source_special02, real_lead_special02, fake_lead_special02 = make_source_special02(one_json_sample)

            if len(source_special02_list) >= batch_size:
                num += 1
                if 'Non' in source_file:
                    # label_list = ["진짜 " + label] * len(source_special01_list)
                    label_list = ["fake"] * len(source_special02_list)

                elif 'Non' not in source_file:
                    # label_list = ["가짜 " + label]  * len(source_special01_list)
                    label_list = ["real"] * len(source_special02_list)

                # source_df = pd.DataFrame({'Title':source_list, 'Content':content_list, 'Label':label_list})
                source_special02_df = pd.DataFrame({'Title':source_special02_list, 'Content':real_lead_special02_list, 'Fake Content':fake_lead_special02_list, 'Label':label_list})
                source_df_path = xlsx_file_path_list[num]
                
                # source_df.to_excel(source_df_path.replace(".xlsx", "_not_special.xlsx"), index=False)
                # source_df.to_csv(source_df_path.replace(".xlsx", "_not_special.csv"), index=False)
                source_special02_df.to_excel(source_df_path.replace(".xlsx", "_special_02.xlsx"), index=False)
                source_special02_df.to_csv(source_df_path.replace(".xlsx", "_special_02.csv"), index=False)

                pbar.n += 1
                pbar.refresh()
                time.sleep(0.01)  

                source_special02_list = []
                real_lead_special02_list = []
                fake_lead_special02_list = []
                    
            elif i == (len(source_file_list) -1): 
                # source_list.append(source)
                source_special02_list.append(source_special02)
                # content_list.append(content)
                real_lead_special02_list.append(real_lead_special02)
                fake_lead_special02_list.append(fake_lead_special02)

                num += 1
                if 'Non' in source_file:
                    # label_list = ["진짜 " + label] * len(source_list)
                    label_list = ["fake"] * len(source_special02_list)

                elif 'Non' not in source_file:
                    # label_list = ["가짜 " + label]  * len(source_list)
                    label_list = ["real"] * len(source_special02_list)

                
                # source_df = pd.DataFrame({'Title':source_list, 'Content':content_list, 'Label':label_list})
                source_special02_df = pd.DataFrame({'Title':source_special02_list, 'Content':real_lead_special02_list, 'Fake Content':fake_lead_special02_list, 'Label':label_list})

                source_df_path = xlsx_file_path_list[num]

                # source_df.to_excel(source_df_path.replace(".xlsx", "_not_special.xlsx"), index=False)
                # source_df.to_csv(source_df_path.replace(".xlsx", "_not_special.csv"), index=False)
                source_special02_df.to_excel(source_df_path.replace(".xlsx", "_special_02.xlsx"), index=False)
                source_special02_df.to_csv(source_df_path.replace(".xlsx", "_special_02.csv"), index=False)

                pbar.n += 1
                pbar.refresh()
                time.sleep(0.01)
                            
            # source_list.append(source)
            source_special02_list.append(source_special02)
            # content_list.append(content)
            real_lead_special02_list.append(real_lead_special02)
            fake_lead_special02_list.append(fake_lead_special02)
        """      
        
    pbar.close()      


def write_spreadsheettext_to_list_merge_file(spreadsheet_folder, dataset_folder, extension):

    spreadsheet_path = glob.glob(spreadsheet_folder +  "*" + extension)
    train_spreadsheet_path = [train_spreadsheet for train_spreadsheet in spreadsheet_path if 'train' in train_spreadsheet ]
    valid_spreadsheet_path = [valid_spreadsheet for valid_spreadsheet in spreadsheet_path if 'valid' in valid_spreadsheet ]

    train_list = []
    valid_list = []
  
    pbar = tqdm(train_spreadsheet_path)
    for load_path in pbar:
        if 'xlsx' in extension: train_spreadsheet = pd.read_excel(load_path, engine='openpyxl')  
        elif 'csv' in extension: train_spreadsheet = pd.read_csv(load_path)  
        titles = train_spreadsheet['Title']
        contents = train_spreadsheet['Content']
        # fake_contents = train_spreadsheet['Fake Content']
        labels = train_spreadsheet['Label']
        for title, content, label in zip(titles, contents, labels):
        # for title, content, fake_content, label in zip(titles, contents, fake_contents, labels):
            # text_label = [title, content, fake_content, label]
            text_label = [title, content, label[2:-2]]
            train_list.append(text_label)
    pbar.close()
   
    pbar = tqdm(valid_spreadsheet_path)
    # valid_titles = []
    # valid_contents = []
    # valid_labels = []

    for load_path in pbar:
        if 'xlsx' in extension: valid_spreadsheet = pd.read_excel(load_path, engine='openpyxl')  
        elif 'csv' in extension: valid_spreadsheet = pd.read_csv(load_path)  
        titles = valid_spreadsheet['Title']
        contents = valid_spreadsheet['Content']
        # fake_contents = valid_spreadsheet['Fake Content']
        # fake_contents = train_spreadsheet['Fake Content']
        labels = valid_spreadsheet['Label']
        for title, content, label in zip(titles, contents, labels):
        # for title, content, fake_content, label in zip(titles, contents, fake_contents, labels):
            # valid_titles.append(title)
            # valid_contents.append(content)
            # valid_labels.append(label)
            # text_label = [title, content, fake_content, label]
            text_label = [title, content, label[2:-2]]
            valid_list.append(text_label)
    pbar.close()
    
    """
    valid_df = pd.DataFrame({"Title":valid_titles, "Content":valid_contents, "Label":valid_labels})
    valid_df['Index'] = valid_df .groupby('Label').cumcount()

    valid_dict = dict(tuple(valid_df.groupby('Index')))

    valid_dict_key = valid_dict.keys()
    valid_dict_value = valid_dict.values()
    random.shuffle(valid_dict_value)
    valid_dict_shuffled = dict(zip(valid_dict_key, valid_dict_value))

    valid_dic1 = dict(list(valid_dict_shuffled.items())[len(valid_dict_shuffled)//2:])
    valid_dic2 = dict(list(valid_dict_shuffled.items())[:len(valid_dict_shuffled)//2])

    valid_df = pd.DataFrame({"Title":['A'], "Content":['A'], "Label":['A']})
    for key, value in valid_dic1.items():
        valid_df = pd.concat([valid_df,value],axis=0)
    valid_df = valid_df.drop(0, axis=0)

    test_df = pd.DataFrame({"Title":['A'], "Content":['A'], "Label":['A']})
    for key, value in valid_dic1.items():
        test_df = pd.concat([test_df,value],axis=0)
    test_df = test_df.drop(0, axis=0)
    """

    # valid_zero = [valid for valid in valid_list if valid[3] == 'fake']
    # valid_one = [valid for valid in valid_list if valid[3] == 'real']
    valid_zero = [valid for valid in valid_list if valid[2] == 'fake']
    valid_one = [valid for valid in valid_list if valid[2] == 'real']


    valid_test_split_zero = int(len(valid_zero) * 0.5)
    valid_zero_copy_list = valid_zero.copy()
    valid_zero_list = valid_zero_copy_list[:valid_test_split_zero]
    test_zero_list = valid_zero_copy_list[valid_test_split_zero:]

    valid_test_split_one = int(len(valid_one) * 0.5)
    valid_one_copy_list = valid_one.copy()
    valid_one_list = valid_one_copy_list[:valid_test_split_one]
    test_one_list = valid_one_copy_list[valid_test_split_one:]

    valid_list = valid_zero_list + valid_one_list
    test_list = test_zero_list + test_one_list
    
    random.shuffle(train_list)
    random.shuffle(valid_list)
    random.shuffle(test_list)

    train_title = [train[0] for train in train_list]
    train_content = [train[1] for train in train_list]
    train_label = [train[2] for train in train_list]
    # train_fake_content = [train[2] for train in train_list]
    # train_label = [train[3] for train in train_list]
    
    valid_title = [valid[0] for valid in valid_list]
    valid_content = [valid[1] for valid in valid_list]
    valid_label = [valid[2] for valid in valid_list]
    # valid_fake_content = [valid[2] for valid in valid_list]
    # valid_label = [valid[3] for valid in valid_list]

    test_title = [test[0] for test in test_list]
    test_content = [test[1] for test in test_list]
    test_label = [test[2] for test in test_list]
    # test_fake_content = [test[2] for test in test_list]
    # test_label = [test[3] for test in test_list]

    # train_df = pd.DataFrame({'Title':train_title, 'Content':train_content, 'Fake Content':train_fake_content, 'Label':train_label}) 
    train_df = pd.DataFrame({'Title':train_title, 'Content':train_content, 'Label':train_label}) 
    train_df_path = dataset_folder + "train_dataset" + extension # "_special.csv" 
    if 'xlsx' in extension: train_df.to_excel(train_df_path, index=False)
    elif 'csv' in extension: train_df.to_csv(train_df_path, index=False)

    # valid_df = pd.DataFrame({'Title':valid_title, 'Content':valid_content, 'Fake Content':valid_fake_content,  'Label':valid_label}) 
    valid_df = pd.DataFrame({'Title':valid_title, 'Content':valid_content, 'Label':valid_label}) 
    valid_df_path = dataset_folder + "valid_dataset" + extension # "_special.csv"
    if 'xlsx' in extension: valid_df.to_excel(valid_df_path, index=False)
    elif 'csv' in extension: valid_df.to_csv(valid_df_path, index=False)

    # test_df = pd.DataFrame({'Title':test_title, 'Content':test_content, 'Fake Content':test_fake_content, 'Label':test_label}) 
    test_df = pd.DataFrame({'Title':test_title, 'Content':test_content, 'Label':test_label}) 
    test_df_path = dataset_folder + "test_dataset" + extension # "_special.csv"
    if 'xlsx' in extension: test_df.to_excel(test_df_path, index=False)
    elif 'csv' in extension: test_df.to_csv(test_df_path, index=False)

    print("The number of train text:", len(train_df))
    print("The number of valid text:", len(valid_df))
    print("The number of test text:", len(test_df))

    print("Train Data:", train_df["Label"].value_counts())
    print()
    print("Valid Data:", valid_df["Label"].value_counts())
    print()
    print("Test Data:", test_df["Label"].value_counts())
    print()
