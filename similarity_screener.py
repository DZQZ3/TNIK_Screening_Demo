from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs

import requests
from io import StringIO
from rdkit.Chem import (
    PandasTools,
    Draw,
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
)
from rdkit.Chem.Draw import MolsToGridImage

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True, parents=True)  # # 确保目录存在，添加parents=True确保父目录也存在


def load_egfr_data():
    """
    加载 EGFR 数据库
    """
    print("=== 加载 EGFR 数据库 ===")

    # EGFR 数据文件路径
    egfr_file = DATA / "EGFR_compounds_lipinski.csv"

    if not egfr_file.exists():
        print(f"错误: 未找到 EGFR 数据文件: {egfr_file}")
        return None

    try:
        # 读取 EGFR 数据
        egfr_data = pd.read_csv(egfr_file)
        print(f"成功加载 EGFR 数据: {len(egfr_data)} 个分子")

        # 检查必要的列
        if 'smiles' not in egfr_data.columns:
            print("错误: EGFR 数据缺少 'smiles' 列")
            return None

        # 添加分子名称列（如果不存在）
        if 'molecule_name' not in egfr_data.columns:
            if 'molecule_chembl_id' in egfr_data.columns:
                egfr_data['molecule_name'] = egfr_data['molecule_chembl_id']
            else:
                egfr_data['molecule_name'] = [f"EGFR_Compound_{i + 1}" for i in range(len(egfr_data))]

        # 添加 RDKit 分子对象
        PandasTools.AddMoleculeColumnToFrame(egfr_data, 'smiles', 'ROMol')

        # 移除无法解析的分子
        original_count = len(egfr_data)
        egfr_data = egfr_data[egfr_data['ROMol'].notnull()]
        filtered_count = len(egfr_data)

        if filtered_count < original_count:
            print(f"过滤掉 {original_count - filtered_count} 个无法解析的分子")

        print("EGFR 分子列表:")
        for i, row in egfr_data.iterrows():
            print(f"{i + 1}. {row['molecule_name']}")

        return egfr_data

    except Exception as e:
        print(f"加载 EGFR 数据失败: {e}")
        return None


def calculate_egfr_similarities(egfr_data, query_smiles):
    """
    计算 EGFR 分子与查询分子的相似度
    """
    print(f"\n=== 计算 EGFR 分子相似度 ===")

    # 创建查询分子
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        print(f"错误: 无法解析查询分子的 SMILES: {query_smiles}")
        return egfr_data

    # 计算查询分子的指纹
    maccs_query = MACCSkeys.GenMACCSKeys(query_mol)

    # 生成 Morgan 指纹生成器
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    morgan_query = fpg.GetFingerprint(query_mol)

    # 为 EGFR 分子计算指纹
    egfr_data["maccs"] = egfr_data["ROMol"].apply(MACCSkeys.GenMACCSKeys)
    egfr_data["morgan"] = egfr_data["ROMol"].apply(fpg.GetFingerprint)

    # 计算相似度
    maccs_list = egfr_data["maccs"].tolist()
    morgan_list = egfr_data["morgan"].tolist()

    egfr_data["tanimoto_maccs"] = DataStructs.BulkTanimotoSimilarity(maccs_query, maccs_list)
    egfr_data["dice_maccs"] = DataStructs.BulkDiceSimilarity(maccs_query, maccs_list)
    egfr_data["tanimoto_morgan"] = DataStructs.BulkTanimotoSimilarity(morgan_query, morgan_list)
    egfr_data["dice_morgan"] = DataStructs.BulkDiceSimilarity(morgan_query, morgan_list)

    # 计算综合相似度
    egfr_data["composite_similarity"] = (
            egfr_data["tanimoto_morgan"] * 0.6 +
            egfr_data["tanimoto_maccs"] * 0.4
    )

    return egfr_data

egfr_data = load_egfr_data()

if egfr_data is not None:
    print("\n使用 EGFR 数据库进行分析")

    # 使用 INS 018_055 作为查询分子
    query_smiles = "CC(C)N1C=NC(=C1C1=NC=C(N1)C(=O)NC1=CC=C(C=C1)N1CCN(C)CC1)C1=CC=C(F)C=C1"

    # 计算相似度
    egfr_data = calculate_egfr_similarities(egfr_data, query_smiles)

    # 重命名列以匹配现有代码
    egfr_data = egfr_data.rename(columns={'molecule_name': 'name'})

    # 使用 EGFR 数据替换原有的 molecules 数据框
    molecules = egfr_data

    print(f"已加载 {len(molecules)} 个 EGFR 分子")
else:
    print("\n使用原有的测试分子进行分析")

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
    molecule = molecules["ROMol"][0]# 选择第8号INS 018_055分子作为示例
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


#MACCS fingerprints: Tanimoto similarity and Dice similarity
print("\n=== MACCS 指纹分子相似性计算 ===")
query_molecule_index = 0# 明确指定查询分子的索引
maccs_query = molecules.loc[query_molecule_index, "maccs"]
maccs_list = molecules["maccs"].tolist()

molecules["tanimoto_maccs"] = DataStructs.BulkTanimotoSimilarity(maccs_query, maccs_list)
molecules["dice_maccs"] = DataStructs.BulkDiceSimilarity(maccs_query, maccs_list)



#Morgan fingerprints: Tanimoto similarity and Dice similarity
print("\n=== Morgan 指纹分子相似性计算 ===")
# Define molecule query and list
query_molecule_index = 0  # 明确指定查询分子的索引
molecule_query = molecules.loc[query_molecule_index, "morgan"]
molecule_list = molecules["morgan"].to_list()

# 计算相似度值
try:
    molecules["tanimoto_morgan"] = DataStructs.BulkTanimotoSimilarity(molecule_query, molecule_list)
    molecules["dice_morgan"] = DataStructs.BulkDiceSimilarity(molecule_query, molecule_list)
except Exception as e:
    print(f"计算相似度时出错: {e}")

# 创建预览数据框，不修改原始数据
preview = (
    molecules
    .sort_values("tanimoto_morgan", ascending=False)
    .reset_index(drop=False)  # 如果不需保留原索引，使用drop=True
    [["name", "tanimoto_morgan", "dice_morgan", "tanimoto_maccs", "dice_maccs"]]
    # 如果确实计算了MACCS相似度，可以添加这些列
    # 否则应该移除对tanimoto_maccs和dice_maccs的引用
)

# 添加格式化的相似度值
preview_display = preview.copy()
preview_display["tanimoto_morgan"] = preview_display["tanimoto_morgan"].apply(lambda x: f"{x:.3f}")
preview_display["dice_morgan"] = preview_display["dice_morgan"].apply(lambda x: f"{x:.3f}")

print("相似度排序结果:")
print(preview_display.to_string(index=False))

# 绘制分子图像
# 生成基于相似度排名的分子网格图像（同时显示Tanimoto和Dice相似度）
try:
    # 按相似度排序
    ranked_molecules = molecules.sort_values("tanimoto_morgan", ascending=False)

    img = Draw.MolsToGridImage(
        ranked_molecules["ROMol"].tolist(),
        molsPerRow=3,  # 每行3个分子
        subImgSize=(450, 150),  # 图像大小
        legends=[
            f"{row['name']}\nTanimoto: {row['tanimoto_morgan']:.3f}\nDice: {row['dice_morgan']:.3f}"
            for _, row in ranked_molecules.iterrows()
        ],
    )

    # 保存图像到文件
    image_path = DATA / "molecule_similarity_ranking_full.png"
    img.save(image_path)
    print(f"相似度排名分子图像已保存至: {image_path}")

except Exception as e:
    print(f"生成相似度排名分子图像时出错: {e}")

#富集因子
def enrichment_analysis_existing(molecules_df, query_index=8):

    print("\n" + "=" * 60)
    print("富集分析 - 基于现有相似度计算结果")


    # 使用已有的相似度列
    if 'tanimoto_morgan' not in molecules_df.columns or 'tanimoto_maccs' not in molecules_df.columns:
        print("错误: 未找到相似度计算结果")
        return

    # 模拟活性分子：基于与查询分子的相似度
    # 使用 Morgan 相似度来定义"活性"分子
    similarity_threshold = 0.6 # 可以调整这个阈值
    active_indices = molecules_df[molecules_df['tanimoto_morgan'] > similarity_threshold].index.tolist()

    print(f"模拟活性分子 (Morgan相似度 > {similarity_threshold}): {len(active_indices)} 个")
    for idx in active_indices:
        name = molecules_df.loc[idx, 'name']
        similarity = molecules_df.loc[idx, 'tanimoto_morgan']
        print(f"  {name}: {similarity:.3f}")

    # 计算富集因子
    fractions = [0.1, 0.2, 0.3, 0.5]  # 10%, 20%, 30%, 50%

    print(f"\n富集因子分析:")

    # MACCS 指纹富集
    print(f"\nMACCS 指纹:")
    maccs_scores = molecules_df['tanimoto_maccs'].values
    for fraction in fractions:
        num_selected = max(1, int(len(molecules_df) * fraction))
        # 从高到低排序
        selected_indices = np.argsort(maccs_scores)[-num_selected:]
        actives_found = len(set(selected_indices) & set(active_indices))
        expected_actives = len(active_indices) * fraction
        enrichment_factor = actives_found / expected_actives if expected_actives > 0 else 0

        print(
            f"  前{fraction * 100:.0f}%: 找到 {actives_found}/{len(active_indices)} 个活性分子, 富集因子 = {enrichment_factor:.2f}")

    # Morgan 指纹富集
    print(f"\nMorgan 指纹:")
    morgan_scores = molecules_df['tanimoto_morgan'].values
    for fraction in fractions:
        num_selected = max(1, int(len(molecules_df) * fraction))
        # 从高到低排序
        selected_indices = np.argsort(morgan_scores)[-num_selected:]
        actives_found = len(set(selected_indices) & set(active_indices))
        expected_actives = len(active_indices) * fraction
        enrichment_factor = actives_found / expected_actives if expected_actives > 0 else 0

        print(
            f"  前{fraction * 100:.0f}%: 找到 {actives_found}/{len(active_indices)} 个活性分子, 富集因子 = {enrichment_factor:.2f}")

    return active_indices


def plot_enrichment_curves_existing(molecules_df, active_indices):
    """
    基于现有数据绘制富集曲线
    """
    print(f"\n=== 生成富集曲线 ===")

    # 计算完整的富集曲线
    fractions = np.linspace(0.01, 1.0, 100)

    # MACCS 富集
    maccs_scores = molecules_df['tanimoto_maccs'].values
    maccs_enrichment = []

    # Morgan 富集
    morgan_scores = molecules_df['tanimoto_morgan'].values
    morgan_enrichment = []

    for fraction in fractions:
        num_selected = max(1, int(len(molecules_df) * fraction))

        # MACCS
        maccs_selected = np.argsort(maccs_scores)[-num_selected:]
        maccs_actives = len(set(maccs_selected) & set(active_indices))
        maccs_expected = len(active_indices) * fraction
        maccs_enrichment.append(maccs_actives / maccs_expected if maccs_expected > 0 else 0)

        # Morgan
        morgan_selected = np.argsort(morgan_scores)[-num_selected:]
        morgan_actives = len(set(morgan_selected) & set(active_indices))
        morgan_expected = len(active_indices) * fraction
        morgan_enrichment.append(morgan_actives / morgan_expected if morgan_expected > 0 else 0)

    # 绘制富集曲线
    plt.figure(figsize=(12, 8))

    plt.plot(fractions * 100, maccs_enrichment, 'b-', linewidth=3, label='MACCS fingerprints')
    plt.plot(fractions * 100, morgan_enrichment, 'r-', linewidth=3, label='Morgan fingerprints')
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Random screening')

    plt.xlabel('Screened Fraction (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Enrichment Factor', fontsize=14, fontweight='bold')
    plt.title('Fingerprint Method Comparison: Enrichment Curves',
              fontsize=16, fontweight='bold')

    # 标注20%处的富集因子
    idx_20 = np.argmin(np.abs(fractions - 0.2))
    maccs_ef_20 = maccs_enrichment[idx_20]
    morgan_ef_20 = morgan_enrichment[idx_20]

    plt.annotate(f'MACCS 20%: EF = {maccs_ef_20:.2f}',
                 xy=(20, maccs_ef_20),
                 xytext=(30, maccs_ef_20 + 0.5),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                 fontsize=12, color='blue', fontweight='bold')

    plt.annotate(f'Morgan 20%: EF = {morgan_ef_20:.2f}',
                 xy=(20, morgan_ef_20),
                 xytext=(30, morgan_ef_20 - 0.5),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=12, color='red', fontweight='bold')

    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)

    # 设置Y轴范围
    max_ef = max(max(maccs_enrichment), max(morgan_enrichment))
    plt.ylim(0, min(max_ef + 0.5, 10))

    plt.tight_layout()
    enrichment_path = DATA / "enrichment_curves_final.png"
    plt.savefig(enrichment_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"富集曲线已保存至: {enrichment_path}")

    return maccs_ef_20, morgan_ef_20


def method_comparison_summary(molecules_df, maccs_ef_20, morgan_ef_20):
    """
    方法比较总结
    """
    print("\n" + "=" * 80)
    print("指纹方法比较总结")


    # 基本统计
    print(f"\n1. 数据集信息:")
    print(f"   总分子数: {len(molecules_df)}")
    print(f"   查询分子: {molecules_df.loc[8, 'name']}")  # INS 018_055

    print(f"\n2. 相似度统计:")
    print(f"   MACCS Tanimoto - 平均值: {molecules_df['tanimoto_maccs'].mean():.3f}, "
          f"标准差: {molecules_df['tanimoto_maccs'].std():.3f}")
    print(f"   Morgan Tanimoto - 平均值: {molecules_df['tanimoto_morgan'].mean():.3f}, "
          f"标准差: {molecules_df['tanimoto_morgan'].std():.3f}")

    print(f"\n3. 20%筛选比例处的富集因子:")
    print(f"   MACCS 指纹: {maccs_ef_20:.2f}")
    print(f"   Morgan 指纹: {morgan_ef_20:.2f}")

    print(f"\n4. 前5个最相似分子 (按Morgan相似度):")

    # 按Morgan相似度排序，排除查询分子本身
    top_similar = molecules_df.nlargest(6, 'tanimoto_morgan')
    # 查找查询分子的索引
    query_mask = top_similar['name'] == 'INS 018_055'
    if query_mask.any():
        top_similar = top_similar[~query_mask].head(5)
    else:
        top_similar = top_similar.head(5)

    for i, (idx, row) in enumerate(top_similar.iterrows(), 1):
        print(f"   {i}. {row['name']}")
        print(f"      Morgan相似度: {row['tanimoto_morgan']:.3f}, "
              f"MACCS相似度: {row['tanimoto_maccs']:.3f}")

    # 判断哪种方法更好 - 修复除以零错误
    if morgan_ef_20 > maccs_ef_20:
        better_method = "Morgan"
        better_ef = morgan_ef_20
        worse_ef = maccs_ef_20
    else:
        better_method = "MACCS"
        better_ef = maccs_ef_20
        worse_ef = morgan_ef_20

    # 修复除以零错误
    if worse_ef > 0:
        improvement = (better_ef - worse_ef) / worse_ef * 100
        improvement_text = f"比另一种方法提高了{improvement:.1f}%"
    else:
        improvement_text = "另一种方法的富集因子为0，无法计算改进百分比"

    print(f"\n5. 结论:")
    print(f"   {better_method}指纹在早期识别方面表现更好")
    print(f"   富集因子{better_ef:.2f}意味着比随机筛选好{better_ef:.1f}倍")
    print(f"   {improvement_text}")



# 1. 基于现有数据进行富集分析
active_indices = enrichment_analysis_existing(molecules, query_index=8)

# 2. 绘制富集曲线
maccs_ef_20, morgan_ef_20 = plot_enrichment_curves_existing(molecules, active_indices)

# 3. 生成方法比较总结
method_comparison_summary(molecules, maccs_ef_20, morgan_ef_20)

# 4. 保存最终结果
final_results_path = DATA / "final_similarity_analysis.csv"
molecules.to_csv(final_results_path, index=False)
print(f"\n最终结果已保存至: {final_results_path}")