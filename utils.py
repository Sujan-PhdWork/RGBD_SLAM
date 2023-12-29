



def data(file_path):

    file_list=[]

    # Specify the file path
    # file_path = '../rgbd_dataset_freiburg1_xyz/depth.txt'

    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:

            for line in file:

                if not line.startswith('#'):
                    words=line.split()
                    file_list.append(words[1])
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    except Exception as e:
        print(f"An error occurred: {e}")

    
    return (file_list)


