from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    PandasTools,
    Draw,
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True, parents=True)  # # 确保目录存在，添加parents=True确保父目录也存在

# Molecules in SMILES format
molecule_smiles = [
    "CC1C2C(C3C(C(=O)C(=C(C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C)O",
    "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
    "C1=COC(=C1)CNC2=CC(=C(C=C2C(=O)O)S(=O)(=O)N)Cl",
    "CCCCCCCCCCCC(=O)OCCOC(=O)CCCCCCCCCCC",
    "C1NC2=CC(=C(C=C2S(=O)(=O)N1)S(=O)(=O)N)Cl",
    "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC(=O)O)C)C",
    "CC1(C2CC3C(C(=O)C(=C(C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C)O",
    "CC1C(CC(=O)C2=C1C=CC=C2O)C(=O)O",
    "CC(C)N1C=NC(=C1C1=NC=C(N1)C(=O)NC1=CC=C(C=C1)N1CCN(C)CC1)C1=CC=C(F)C=C1",
]

# List of molecule names
molecule_names = [
    "Doxycycline",
    "Amoxicilline",
    "Furosemide",
    "Glycol dilaurate",
    "Hydrochlorothiazide",
    "Isotretinoin",
    "Tetracycline",
    "Hemi-cycline D",
    "INS 018_055",
]

# 创建分子数据框
print("=== 创建分子数据框 ===")
molecules = pd.DataFrame({"smiles": molecule_smiles, "name": molecule_names})

try:
    PandasTools.AddMoleculeColumnToFrame(molecules, smilesCol="smiles")
except Exception as e:
    print(f"添加分子列时出错: {e}")
    # 如果添加分子列失败，尝试手动创建ROMol列
    molecules["ROMol"] = molecules["smiles"].apply(lambda x: Chem.MolFromSmiles(x) if x else None)


print("=== 分子信息表 ===")
print(molecules[['name', 'smiles']].to_string())  # 名称和SMILES

print("\n=== 生成分子图像 ===")
# 生成分子网格图像
try:
    img = Draw.MolsToGridImage(
        molecules["ROMol"].to_list(),
        molsPerRow=4,  # 4个一行显示
        subImgSize=(300, 150),
        legends=molecules["name"].to_list(),
        returnPNG=False  # 确保返回的是图像对象而不是PNG数据
    )

    # 保存图像到文件
    image_path = DATA / "molecule_grid.png"
    img.save(image_path)
    print(f"分子图像已保存至: {image_path}")

except Exception as e:
    print(f"生成分子图像时出错: {e}")


# 一維分子描述符：分子量
print("\n=== 计算分子量 ===")
molecules["molecule_weight"] = molecules.ROMol.apply(Descriptors.MolWt)
# Sort molecules by molecular weight, 修改原数据框
molecules.sort_values(["molecule_weight"], ascending=False, inplace=True)

# Show only molecule names and molecular weights
print(molecules[["smiles", "name", "molecule_weight"]])

print("\n=== 生成排序后的分子图像 ===")
# 生成分子网格图像
try:
    img = Draw.MolsToGridImage(
        molecules["ROMol"].tolist(),
        molsPerRow=3,  # 保持原来的3个一行
        subImgSize=(450, 150),  # 保持原来的图像大小
        legends=[
            f"{row['name']}: {row['molecule_weight']:.2f} Da"
            for _, row in molecules.iterrows()
        ],
    )

    # 保存图像到文件
    image_path = DATA / "molecule_grid1.png"
    img.save(image_path)
    print(f"分子图像已保存至: {image_path}")

except Exception as e:
    print(f"生成排序后分子图像时出错: {e}")


# 二維分子描述子：MACCS指紋
print("\n=== 生成MACCS指纹 ===")
try:
    molecule = molecules["ROMol"][0]# 选择第0号分子作为示例
    print(f"示例分子: {molecules['name'][0]}")

    # 生成MACCS指紋
    maccs_fp = MACCSkeys.GenMACCSKeys(molecule)
    print("MACCS指紋位元串:")
    print(maccs_fp.ToBitString())

    # 應用於所有分子：將所有分子的MACCS指紋添加到DataFrame
    molecules["maccs"] = molecules.ROMol.apply(MACCSkeys.GenMACCSKeys)
    print("已為所有分子添加MACCS指紋")

except Exception as e:
    print(f"生成MACCS指纹时出错: {e}")


# 二維分子描述子：摩根指紋
print("\n=== 生成摩根指纹 ===")
try:
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # 將Morgan指紋產生為整數向量
    circular_int_fp = fpg.GetCountFingerprint(molecule)
    print(f"打印非零元素:\n{circular_int_fp.GetNonzeroElements()}")

    # 產生摩根指紋作為位元向量
    circular_bit_fp = fpg.GetFingerprint(molecule)
    print(f"打印前100個指紋位:\n{circular_bit_fp.ToBitString()[:100]}") # 只显示前100位

    # 應用於所有分子
    molecules["morgan"] = molecules["ROMol"].map(fpg.GetFingerprint)

except Exception as e:
    print(f"生成摩根指纹时出错: {e}")

print("\n=== 分子相似性计算 ===")
#范例分子对：将两个 MACCS 指纹与 Tanimoto 相似性进行比较
# Example molecules
molecule1 = molecules["ROMol"][0]
molecule2 = molecules["ROMol"][1]

# Example fingerprints
maccs_fp1 = MACCSkeys.GenMACCSKeys(molecule1)
maccs_fp2 = MACCSkeys.GenMACCSKeys(molecule2)

#计算两个不同分子之间的Tanimoto系数
DataStructs.TanimotoSimilarity(maccs_fp1, maccs_fp2)
#计算两个相同分子之间的Tanimoto系数, 这个结果应是1
DataStructs.TanimotoSimilarity(maccs_fp1, maccs_fp1)

#test