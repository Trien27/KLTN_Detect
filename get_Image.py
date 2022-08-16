from longthanhapi import driveapi
from datetime import datetime
import glob

datenow = str(datetime.today().strftime('%d-%m-%Y'))
services_start = driveapi('/home/xavier/Google_Drive_API_Basic/code_secret_client.json', 'drive', 'v3', 'https://www.googleapis.com/auth/drive')
service = services_start.service_api()
main_folder = services_start.create_folder(service, datenow)
task_name = ['HAR', 'PPE']
folder_in_parents = services_start.create_folder_in_folder(service, 1, main_folder, task_name)
list_have_har=[]
list_have_ppe=[]
while True:
    datenow = str(datetime.today().strftime('%d-%m-%Y'))
    date_img = str(datetime.today().strftime('%Y-%m-%d'))
    token_access = services_start.token_access_get()
    check = services_start.check_exist_folder(datenow, token_access)
    har = folder_in_parents[0] 
    print(har)
    ppe = folder_in_parents[1]
    print(ppe)
    if check == 0: # folder not existing
        print('Task 1')
        main_folder = services_start.create_folder(service, datenow)
        folder_in_parents = services_start.create_folder_in_folder(service, 1, main_folder, task_name)
    elif check == 1: # folder existing
        print('Task 2')
        for filename in glob.glob('/home/xavier/Cloud/FALL/{}/*.jpg'.format(date_img)):
            get_Name=filename[35:]
            if get_Name not in list_have_har:
               image_upload = services_start.upload_image(filename, service, get_Name, har)
               list_have_har.append(get_Name)
        for filename1 in glob.glob('/home/xavier/Cloud/PPE/{}/*.jpg'.format(date_img)):
            get_Name1=filename1[36:]
            if get_Name1 not in list_have_ppe:
               image_upload1 = services_start.upload_image(filename1, service, get_Name1, ppe)
               list_have_ppe.append(get_Name1)
        







