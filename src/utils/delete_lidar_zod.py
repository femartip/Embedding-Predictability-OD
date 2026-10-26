import os 
import shutil

if __name__ == "__main__":
    zod_path = "./data/zod/single_frames"

    id_dir = os.listdir(zod_path)

    for dir in id_dir:
        global_path = os.path.join(zod_path, dir)
        if os.path.isdir(global_path):
          lidar_folder = os.path.join(global_path, "lidar_velodyne")  
          if os.path.isdir(lidar_folder):
            shutil.rmtree(lidar_folder)
            print(f"Lidar data in {dir} removed")
          else:
             print(f"No lidar data in {dir}")  

    print("Finished")

