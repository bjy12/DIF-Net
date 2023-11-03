import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk

def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_img)
    return image


def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)


def generate_blocks():
    block_list = []
    base = np.mgrid[:64, :64, :64] * 4 # 3, 64 ^ 3
    #* np.mgrid 可视化
    #base1 = np.mgrid[:3 , :3 , :3]
    #print( " base 1 : " , base1)
    #base2 = base1.reshape(3 , -1 )
    #print( " base2 : " , base2)
    #x , y , z = base1
    #test_base = base1[0 , : ,: ,:]
    #print("test_base " , test_base)
    #test_base2 = np.mgrid[:3 , :3 ]
    #test_base2 = test_base2.reshape(3,3,-1)
    #print( " test_base2 : " , test_base2.shape)

    #print( "x " , x)
    #print( "y " , y)
    #print( "z " , z)
    #x = x.flatten()
    #y = y.flatten()
    #z = z.flatten()

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    #ax.scatter(x, y, z)

    # 设置图形标签
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')

    # 显示图形
    #plt.show()
    #print(" base1 shape : " , base1.shape)
    print(" base shape " , base.shape)
    #构建了一个 64 * 64 *64 的网格 每个网格之间的距离为4
    base = base.reshape(3, -1)
    print( " base reshape : " , base.shape)
    print(" base reshape : " , base)
    #* -------------Test
    #offset = np.array([0,0,2])
    #block = base + offset[: , None]
    #print( " block : " , block.shape)
    #print( " block : " , block)
    for x in range(4):
        for y in range(4):
            for z in range(4):
                offset = np.array([x, y, z])
                block = base + offset[:, None]
                block_list.append(block)
    print( " block_list : " , len(block_list))
    return block_list


if __name__ == '__main__':
    os.makedirs('./blocks/', exist_ok=True)
    #创建一个block_list来代表一个ct 每个block 是每个体素中的一小块
    block_list = generate_blocks()
    np_block_list = np.array(block_list)
    print(" np_block_list : " , np_block_list.shape)
    # np.stack 堆叠 按着 block_list 内块的个数进行堆叠形成一个 64 ， 3 ， 262144（64*64*64）
    blocks = np.stack(block_list, axis=0) # K, 3, N^3
    print( " blocks : " , blocks.shape)
    blocks = blocks.transpose(0, 2, 1).astype(float) / 255 # K, N^3, 3
    print( " blocks shape: " , blocks.shape)
    #print( " blocks: " , blocks)
    np.save('./blocks/blocks.npy', blocks)
    files = glob(f'processed/*.nii.gz')
    print("files " , files)
    for file in tqdm(files, ncols=50):
        name = file.split('\\')[-1].split('.')[0]
        print(" name : " , name)
        data_path = f'./processed/{name}.nii.gz'
        image = read_nifti(data_path)
    
        save_dir = f'./blocks/{name}/'
        os.makedirs(save_dir, exist_ok=True)
        #* block 是每个CT_Scan保存为64 个块 ，64个块代表整体的像素 
        for k, block in enumerate(block_list):
            block = block.reshape(3, -1).transpose(1, 0)
            image_block = image[block[:, 0], block[:, 1], block[:, 2]]
            np.save(os.path.join(save_dir, f'block_{k}.npy'), image_block)
