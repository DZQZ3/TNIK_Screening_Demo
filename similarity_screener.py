from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

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
DATA.mkdir(exist_ok=True, parents=True)


def load_specs_cleanD008_data():
    """
    加载 Specs_CleanD008 数据库
    """
    print("=== 加载 Specs_CleanD008 数据库 ===")

    # Specs_CleanD008 数据文件路径
    specs_file = DATA / "specs_cleanD008.csv"

    if not specs_file.exists():
        print(f"错误: 未找到 Specs_CleanD008 数据文件: {specs_file}")
        return None

    try:
        # 尝试不同的编码方式读取CSV文件
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        specs_data = None

        for encoding in encodings:
            try:
                specs_data = pd.read_csv(specs_file, encoding=encoding)
                print(f"使用 {encoding} 编码成功读取文件")
                break
            except UnicodeDecodeError:
                continue

        if specs_data is None:
            print("错误: 无法读取文件，尝试所有编码都失败")
            return None

        print(f"成功加载 Specs_CleanD008 数据: {len(specs_data)} 个分子")
        print(f"数据列: {list(specs_data.columns)}")

        # 查找 SMILES 列
        smiles_col = None
        possible_smiles_cols = ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'structure',
                                'SMILES_CANONICAL', 'Canonical_Smiles', 'rdkit_smiles']

        for col in possible_smiles_cols:
            if col in specs_data.columns:
                smiles_col = col
                break

        if smiles_col is None:
            print("错误: 未找到 SMILES 列")
            return None

        print(f"使用 SMILES 列: {smiles_col}")

        # 添加分子名称列（如果不存在）
        if 'molecule_name' not in specs_data.columns:
            if 'compound_name' in specs_data.columns:
                specs_data['molecule_name'] = specs_data['compound_name']
            elif 'Name' in specs_data.columns:
                specs_data['molecule_name'] = specs_data['Name']
            elif 'ID' in specs_data.columns:
                specs_data['molecule_name'] = specs_data['ID']
            else:
                specs_data['molecule_name'] = [f"Specs_Compound_{i + 1}" for i in range(len(specs_data))]

        # 重命名SMILES列为标准名称
        specs_data['smiles'] = specs_data[smiles_col]

        # 添加 RDKit 分子对象
        print("添加分子结构...")
        PandasTools.AddMoleculeColumnToFrame(specs_data, 'smiles', 'ROMol')

        # 移除无法解析的分子
        original_count = len(specs_data)
        specs_data = specs_data[specs_data['ROMol'].notnull()]
        filtered_count = len(specs_data)

        if filtered_count < original_count:
            print(f"过滤掉 {original_count - filtered_count} 个无法解析的分子")

        return specs_data

    except Exception as e:
        print(f"加载 Specs_CleanD008 数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_fingerprint_similarities(data, query_smiles, query_name="INS 018_055"):
    """
    计算三种指纹的相似度: MACCS, Morgan, RDKit拓扑指纹
    """
    print(f"\n=== 计算分子相似度 - 查询分子: {query_name} ===")

    # 创建查询分子
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        print(f"错误: 无法解析查询分子的 SMILES: {query_smiles}")
        return data

    # 计算查询分子的三种指纹
    print("生成查询分子的三种指纹中")
    maccs_query = MACCSkeys.GenMACCSKeys(query_mol)

    # Morgan 指纹
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    morgan_query = fpg.GetFingerprint(query_mol)

    # RDKit 拓扑指纹
    rdkit_query = Chem.RDKFingerprint(query_mol)

    # 为所有分子计算三种指纹
    print("为数据库分子计算指纹中")
    data["maccs"] = data["ROMol"].apply(MACCSkeys.GenMACCSKeys)
    data["morgan"] = data["ROMol"].apply(fpg.GetFingerprint)
    data["rdkit"] = data["ROMol"].apply(Chem.RDKFingerprint)

    # 计算三种指纹的相似度
    print("计算相似度中")

    # MACCS
    maccs_list = data["maccs"].tolist()
    data["tanimoto_maccs"] = DataStructs.BulkTanimotoSimilarity(maccs_query, maccs_list)
    data["dice_maccs"] = DataStructs.BulkDiceSimilarity(maccs_query, maccs_list)

    # Morgan
    morgan_list = data["morgan"].tolist()
    data["tanimoto_morgan"] = DataStructs.BulkTanimotoSimilarity(morgan_query, morgan_list)
    data["dice_morgan"] = DataStructs.BulkDiceSimilarity(morgan_query, morgan_list)

    # RDKit 拓扑指纹
    rdkit_list = data["rdkit"].tolist()
    data["tanimoto_rdkit"] = DataStructs.BulkTanimotoSimilarity(rdkit_query, rdkit_list)
    data["dice_rdkit"] = DataStructs.BulkDiceSimilarity(rdkit_query, rdkit_list)

    # 计算综合相似度 (三种指纹的加权平均)
    data["composite_similarity"] = (
            data["tanimoto_morgan"] * 0.4 +
            data["tanimoto_maccs"] * 0.3 +
            data["tanimoto_rdkit"] * 0.3
    )

    print("指纹相似度计算完成!")
    return data


# =============================================================================
# 主程序开始
# =============================================================================

print("TNIK Inhibitor Similarity Screening - 三种指纹方法比较")
print("=" * 60)

# 首先尝试加载 Specs_CleanD008 数据库
specs_data = load_specs_cleanD008_data()

if specs_data is not None:
    print("\n使用 Specs_CleanD008 数据库进行分析")

    # 使用 INS 018_055 作为查询分子
    query_smiles = "CC(C)N1C=NC(=C1C1=NC=C(N1)C(=O)NC1=CC=C(C=C1)N1CCN(C)CC1)C1=CC=C(F)C=C1"
    query_name = "INS 018_055"

    # 计算三种指纹的相似度
    specs_data = calculate_fingerprint_similarities(specs_data, query_smiles, query_name)

    # 重命名列以匹配后续代码
    specs_data = specs_data.rename(columns={'molecule_name': 'name'})

    # 使用 Specs 数据
    molecules = specs_data
    print(f"已加载 {len(molecules)} 个 Specs 分子")

else:
    print("\nSpecs_CleanD008 数据库未找到或无法读取")



# =============================================================================
# 可视化部分
# =============================================================================

# print("\n=== 生成分子图像 ===")
# try:
#     # 按综合相似度排序
#     ranked_molecules = molecules.sort_values("composite_similarity", ascending=False)
#
#     # 只显示前12个分子以避免图像过大
#     display_molecules = ranked_molecules.head(12)
#
#     img = Draw.MolsToGridImage(
#         display_molecules["ROMol"].tolist(),
#         molsPerRow=4,
#         subImgSize=(300, 200),
#         legends=[
#             f"{row['name']}\nMorgan: {row['tanimoto_morgan']:.3f}\nMACCS: {row['tanimoto_maccs']:.3f}\nRDKit: {row['tanimoto_rdkit']:.3f}"
#             for _, row in display_molecules.iterrows()
#         ],
#     )
#
#     image_path = DATA / "specs_similarity_grid.png"
#     img.save(image_path)
#     print(f"分子图像已保存至: {image_path}")
#
# except Exception as e:
#     print(f"生成分子图像时出错: {e}")


# =============================================================================
# 富集分析部分 - 三种指纹比较
# =============================================================================

def enrichment_analysis_three_fingerprints(molecules_df, query_index=0):
    """
    三种指纹的富集分析
    """
    print("\n" + "=" * 60)
    print("富集分析 - 三种指纹方法比较")
    print("=" * 60)

    # 模拟活性分子：基于与查询分子的综合相似度
    # 使用分位数来定义活性分子
    if len(molecules_df) > 20:
        similarity_threshold = molecules_df['composite_similarity'].quantile(0.8)  # 前20%
    else:
        similarity_threshold = 0.6  # 可以调整这个阈值

    active_indices = molecules_df[molecules_df['composite_similarity'] > similarity_threshold].index.tolist()

    print(f"模拟活性分子 (综合相似度 > {similarity_threshold:.3f}): {len(active_indices)} 个")
    if len(active_indices) > 0:
        for idx in active_indices[:5]:  # 只显示前5个
            name = molecules_df.loc[idx, 'name']
            similarity = molecules_df.loc[idx, 'composite_similarity']
            print(f"  {name}: {similarity:.3f}")
    else:
        print("  未找到活性分子，调整阈值中")
        # 如果没有找到活性分子，使用前10%作为活性分子
        active_indices = molecules_df.nlargest(max(1, len(molecules_df) // 10), 'composite_similarity').index.tolist()
        print(f"  使用前10%作为活性分子: {len(active_indices)} 个")

    # 计算富集因子
    fractions = [0.1, 0.2, 0.3, 0.5]  # 10%, 20%, 30%, 50%

    print(f"\n富集因子分析:")

    # 三种指纹的富集分析
    fingerprint_methods = {
        'MACCS': 'tanimoto_maccs',
        'Morgan': 'tanimoto_morgan',
        'RDKit': 'tanimoto_rdkit'
    }

    enrichment_results = {}

    for method_name, score_col in fingerprint_methods.items():
        print(f"\n{method_name} 指纹:")
        scores = molecules_df[score_col].values

        method_results = []
        for fraction in fractions:
            num_selected = max(1, int(len(molecules_df) * fraction))
            # 从高到低排序
            selected_indices = np.argsort(scores)[-num_selected:]
            actives_found = len(set(selected_indices) & set(active_indices))
            expected_actives = len(active_indices) * fraction
            enrichment_factor = actives_found / expected_actives if expected_actives > 0 else 0

            method_results.append(enrichment_factor)
            print(
                f"  前{fraction * 100:.0f}%: 找到 {actives_found}/{len(active_indices)} 个活性分子, 富集因子 = {enrichment_factor:.2f}")

        enrichment_results[method_name] = method_results

    return active_indices, enrichment_results


def plot_enrichment_curves_three_methods(molecules_df, active_indices):
    """
    绘制三种指纹的富集曲线
    """
    print(f"\n=== 生成三种指纹的富集曲线 ===")

    # 计算完整的富集曲线
    fractions = np.linspace(0.01, 1.0, 100)

    # 三种指纹的富集数据
    fingerprint_methods = {
        'MACCS': 'tanimoto_maccs',
        'Morgan': 'tanimoto_morgan',
        'RDKit': 'tanimoto_rdkit'
    }

    enrichment_data = {}
    ef_20_values = {}

    for method_name, score_col in fingerprint_methods.items():
        scores = molecules_df[score_col].values
        method_enrichment = []

        for fraction in fractions:
            num_selected = max(1, int(len(molecules_df) * fraction))
            selected_indices = np.argsort(scores)[-num_selected:]
            actives_found = len(set(selected_indices) & set(active_indices))
            expected_actives = len(active_indices) * fraction
            enrichment_factor = actives_found / expected_actives if expected_actives > 0 else 0
            method_enrichment.append(enrichment_factor)

        enrichment_data[method_name] = method_enrichment

        # 记录20%处的富集因子
        idx_20 = np.argmin(np.abs(fractions - 0.2))
        ef_20_values[method_name] = method_enrichment[idx_20]

    # 绘制富集曲线
    plt.figure(figsize=(12, 8))

    colors = {'MACCS': 'blue', 'Morgan': 'red', 'RDKit': 'green'}

    for method_name, enrichment_values in enrichment_data.items():
        plt.plot(fractions * 100, enrichment_values,
                 color=colors[method_name], linewidth=3,
                 label=f'{method_name} fingerprints')

    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Random screening')

    plt.xlabel('Screened Fraction (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Enrichment Factor', fontsize=14, fontweight='bold')
    plt.title('Specs_CleanD008: Three Fingerprint Methods Comparison\nEnrichment Curves',
              fontsize=16, fontweight='bold')

    # 标注20%处的富集因子
    y_positions = [0.5, 0, -0.5]  # 三个标注的垂直位置
    for i, (method_name, ef_20) in enumerate(ef_20_values.items()):
        plt.annotate(f'{method_name} 20%: EF = {ef_20:.2f}',
                     xy=(20, ef_20),
                     xytext=(30, ef_20 + y_positions[i]),
                     arrowprops=dict(arrowstyle='->', color=colors[method_name], lw=2),
                     fontsize=11, color=colors[method_name], fontweight='bold')

    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)

    # 设置Y轴范围
    max_ef = max(max(values) for values in enrichment_data.values())
    plt.ylim(0, min(max_ef + 0.5, 10))

    plt.tight_layout()
    enrichment_path = DATA / "specs_three_methods_enrichment_curves.png"
    plt.savefig(enrichment_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"三种指纹富集曲线已保存至: {enrichment_path}")

    return ef_20_values


def three_methods_comparison_summary(molecules_df, ef_20_values):
    """
    三种指纹方法的比较总结
    """
    print("\n" + "=" * 80)
    print("Specs_CleanD008 数据库 - 三种指纹方法比较总结")
    print("=" * 80)

    # 基本统计
    print(f"\n1. 数据集信息:")
    print(f"   总分子数: {len(molecules_df)}")
    print(f"   查询分子: INS 018_055 (TNIK抑制剂)")

    print(f"\n2. 相似度统计:")
    for method in ['maccs', 'morgan', 'rdkit']:
        col_name = f'tanimoto_{method}'
        if col_name in molecules_df.columns:
            mean_val = molecules_df[col_name].mean()
            std_val = molecules_df[col_name].std()
            max_val = molecules_df[col_name].max()
            print(
                f"   {method.upper()} Tanimoto - 平均值: {mean_val:.3f}, 标准差: {std_val:.3f}, 最大值: {max_val:.3f}")

    print(f"\n3. 20%筛选比例处的富集因子:")
    best_method = None
    best_ef = 0
    for method, ef in ef_20_values.items():
        print(f"   {method} 指纹: {ef:.2f}")
        if ef > best_ef:
            best_ef = ef
            best_method = method

    print(f"\n4. 前5个最相似分子 (按综合相似度):")
    top_similar = molecules_df.nlargest(5, 'composite_similarity')

    for i, (idx, row) in enumerate(top_similar.iterrows(), 1):
        print(f"   {i}. {row['name']}")
        print(f"      综合相似度: {row['composite_similarity']:.3f}")
        print(
            f"      Morgan: {row['tanimoto_morgan']:.3f}, MACCS: {row['tanimoto_maccs']:.3f}, RDKit: {row['tanimoto_rdkit']:.3f}")
    '''
    print(f"\n5. 结论:")
    print(f"   {best_method}指纹在早期识别方面表现最好")
    print(f"   富集因子{best_ef:.2f}意味着比随机筛选好{best_ef:.1f}倍")

    # 计算改进百分比
    other_methods = [m for m in ef_20_values.keys() if m != best_method]
    if other_methods and all(ef_20_values[m] > 0 for m in other_methods):
        avg_other_ef = np.mean([ef_20_values[m] for m in other_methods])
        improvement = (best_ef - avg_other_ef) / avg_other_ef * 100
        print(f"   比其他指纹方法平均提高了{improvement:.1f}%")

    print(f"\n6. 建议:")
    print(f"   对于TNIK抑制剂的虚拟筛选，建议使用{best_method}指纹方法")
    print(f"   因为它在早期筛选阶段提供最好的富集效果")

    # 额外建议
    print(f"\n7. 数据库质量评估:")
    max_similarity = molecules_df['composite_similarity'].max()
    if max_similarity > 0.7:
        print(f"   数据库中存在高度相似的分子 (最高相似度: {max_similarity:.3f})")
        print(f"   数据库适合用于基于相似性的虚拟筛选")
    elif max_similarity > 0.4:
        print(f"   数据库中分子相似度中等 (最高相似度: {max_similarity:.3f})")
        print(f"   可能需要结合其他虚拟筛选方法")
    else:
        print(f"   数据库中分子相似度较低 (最高相似度: {max_similarity:.3f})")
        print(f"   建议寻找更相关的化合物库")
    '''
    print("=" * 80)


# =============================================================================
# 执行分析
# =============================================================================

print("\n开始富集分析...")

# 1. 三种指纹的富集分析
active_indices, enrichment_results = enrichment_analysis_three_fingerprints(molecules)

# 2. 绘制三种指纹的富集曲线
ef_20_values = plot_enrichment_curves_three_methods(molecules, active_indices)

# 3. 生成三种方法的比较总结
three_methods_comparison_summary(molecules, ef_20_values)

# 4. 保存最终结果
final_results_path = DATA / "specs_three_fingerprints_analysis_results.csv"
# 只保存重要的列
important_columns = ['name', 'smiles', 'composite_similarity',
                     'tanimoto_morgan', 'tanimoto_maccs', 'tanimoto_rdkit',
                     'dice_morgan', 'dice_maccs', 'dice_rdkit']

available_columns = [col for col in important_columns if col in molecules.columns]
molecules[available_columns].to_csv(final_results_path, index=False)

print(f"\n最终结果已保存至: {final_results_path}")
print("\n=== 分析完成 ===")