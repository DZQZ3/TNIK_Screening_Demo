# deduplicate_for_docking.py
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True, parents=True)


def load_similarity_results():

    print("=== 加载相似度分析结果 ===")

    results_file = DATA / "specs_three_fingerprints_analysis_results.csv"

    if not results_file.exists():
        print(f"错误: 未找到相似度分析结果文件: {results_file}")
        print("请先运行 similarity_screener.py 生成相似度数据")
        return None

    try:
        results_data = pd.read_csv(results_file)
        print(f"成功加载相似度数据: {len(results_data)} 个分子")

        # 备份原始文件（未去重）
        backup_file = DATA / "specs_three_fingerprints_analysis_results_original.csv"
        results_data.to_csv(backup_file, index=False)
        print(f"原始数据已备份: {backup_file}")

        return results_data
    except Exception as e:
        print(f"加载相似度数据失败: {e}")
        return None


def filter_high_similarity_molecules(data):
    """
    筛选三种指纹相似度 > 60% 的分子
    """
    print("\n=== 筛选高相似度分子 ===")

    # 相似度 > 60%
    similarity_threshold = 0.6


    morgan_condition = data['tanimoto_morgan'] > similarity_threshold
    maccs_condition = data['tanimoto_maccs'] > similarity_threshold
    rdkit_condition = data['tanimoto_rdkit'] > similarity_threshold

    high_similarity_mask = morgan_condition | maccs_condition | rdkit_condition

    filtered_data = data[high_similarity_mask].copy()

    print(f"筛选条件: 任一指纹相似度 > {similarity_threshold * 100}%")
    print(f"原始分子数: {len(data)}")
    print(f"筛选后分子数: {len(filtered_data)}")

    # 显示筛选统计
    print(f"\n各指纹满足条件的分子数:")
    print(f"  Morgan > 60%: {morgan_condition.sum()}")
    print(f"  MACCS > 60%: {maccs_condition.sum()}")
    print(f"  RDKit > 60%: {rdkit_condition.sum()}")

    return filtered_data


def remove_duplicates_by_smiles(data):
    """
    基于SMILES进行去重，保留最高综合相似度的分子
    """
    print("\n=== 基于SMILES去重 ===")

    # 按综合相似度降序排序，这样在去重时会保留相似度最高的
    data_sorted = data.sort_values('composite_similarity', ascending=False)

    # 基于规范的SMILES去重
    print("生成规范SMILES")
    data_sorted['canonical_smiles'] = data_sorted['smiles'].apply(
        lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if pd.notna(x) and x != '' else x
    )

    # 去除重复的SMILES，保留第一个（即综合相似度最高的）
    original_count = len(data_sorted)
    deduplicated_data = data_sorted.drop_duplicates(subset=['canonical_smiles'], keep='first')
    final_count = len(deduplicated_data)

    print(f"去重前分子数: {original_count}")
    print(f"去重后分子数: {final_count}")
    print(f"去除重复分子: {original_count - final_count}")

    return deduplicated_data


def prepare_docking_files(data):
    """
    准备分子对接所需的文件
    """
    print("\n=== 准备分子对接文件 ===")

    # 1. 生成SDF文件 (Schrödinger Maestro需要)
    print("生成SDF文件")
    sdf_file = DATA / "docking_candidates.sdf"

    # 添加RDKit分子对象
    PandasTools.AddMoleculeColumnToFrame(data, 'smiles', 'ROMol')

    # 保存为SDF
    PandasTools.WriteSDF(data, str(sdf_file), molColName='ROMol',
                         properties=list(data.columns))
    print(f"SDF文件已保存: {sdf_file}")

    # 2. 生成简化CSV文件（仅关键信息）
    simplified_csv = DATA / "docking_candidates_simplified.csv"
    simplified_data = data[['name', 'smiles', 'canonical_smiles',
                            'composite_similarity', 'tanimoto_morgan',
                            'tanimoto_maccs', 'tanimoto_rdkit']].copy()
    simplified_data.to_csv(simplified_csv, index=False)
    print(f"简化CSV文件已保存: {simplified_csv}")

    return sdf_file, simplified_csv



def main():

    print("TNIK抑制剂候选分子去重程序\n")


    # 1. 加载相似度结果
    similarity_data = load_similarity_results()
    if similarity_data is None:
        return

    # 2. 筛选高相似度分子
    high_similarity_data = filter_high_similarity_molecules(similarity_data)

    if len(high_similarity_data) == 0:
        print("没有找到相似度 > 60% 的分子")
        print("请调整筛选阈值或检查原始数据")
        return

    # 3. 基于SMILES去重
    deduplicated_data = remove_duplicates_by_smiles(high_similarity_data)

    # 4. 准备对接文件
    sdf_file, csv_file = prepare_docking_files(deduplicated_data)




    print("去重程序完成")
    print("=" * 50)
    print(f"生成的对接候选分子数: {len(deduplicated_data)}")
    print(f"主要输出文件:")
    print(f"  对接分子文件: {sdf_file}")
    print(f"  简化数据文件: {csv_file}")
    print(f"  原始数据备份: specs_three_fingerprints_analysis_results_original.csv")



if __name__ == "__main__":
    main()