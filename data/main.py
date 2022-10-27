from wos_data_core_ref_txt import *
main_path = r'D:\实验室电脑\SME1018\SME\data\demo'
email = '374058832@qq.com'
df_core_wos = convert_dict_to_df(extract_doi_kvlist_from_txt(main_path,histcite = False))

