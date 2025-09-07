import csv
import json
from typing import Dict, Tuple, Optional

class PromptLengthAnalyzer:
    def __init__(self, csv_file_path: str):
        """
        初始化Prompt长度分析器
        
        Args:
            csv_file_path: CSV文件路径
        """
        self.csv_file_path = csv_file_path
        self.length_data = self._load_csv_data()
    
    def _load_csv_data(self) -> Dict[str, Dict]:
        """
        加载CSV数据到内存
        
        Returns:
            以问题为键，包含token_count和length_category的字典
        """
        data = {}
        try:
            with open(self.csv_file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data[row['question']] = {
                        'token_count': int(row['token_count']),
                        'length_category': row['length_category'],
                        'context': row['context']
                    }
            print(f"成功加载 {len(data)} 条问题数据")
        except FileNotFoundError:
            print(f"警告: CSV文件 {self.csv_file_path} 未找到")
            data = {}
        except Exception as e:
            print(f"加载CSV文件时出错: {e}")
            data = {}
        
        return data
    
    def reload_data(self) -> None:
        """重新加载CSV数据"""
        self.length_data = self._load_csv_data()
    
    def get_length_info(self, question: str) -> Optional[Dict[str, any]]:
        """
        根据问题获取长度信息
        
        Args:
            question: 问题文本
            
        Returns:
            包含token_count和length_category的字典，如果问题不存在则返回None
        """
        return self.length_data.get(question)
    
    def get_token_count(self, question: str) -> Optional[int]:
        """
        获取问题的token数量
        
        Args:
            question: 问题文本
            
        Returns:
            token数量，如果问题不存在则返回None
        """
        info = self.get_length_info(question)
        return info['token_count'] if info else None
    
    def get_length_category(self, question: str) -> Optional[str]:
        """
        获取问题的长度类别
        
        Args:
            question: 问题文本
            
        Returns:
            长度类别字符串，如果问题不存在则返回None
        """
        info = self.get_length_info(question)
        return info['length_category'] if info else None
    
    def get_all_questions_in_category(self, category: str) -> list:
        """
        获取指定长度类别的所有问题
        
        Args:
            category: 长度类别（如"0k-8k", "8k-16k"等）
            
        Returns:
            属于该类别的问题列表
        """
        return [q for q, info in self.length_data.items() if info['length_category'] == category]
    
    def get_category_statistics(self) -> Dict[str, int]:
        """
        获取各长度类别的统计信息
        
        Returns:
            各长度类别的问题数量统计
        """
        statistics = {}
        for info in self.length_data.values():
            category = info['length_category']
            statistics[category] = statistics.get(category, 0) + 1
        return statistics
    
    def print_category_statistics(self) -> None:
        """打印长度类别统计信息"""
        stats = self.get_category_statistics()
        print("\n=== 长度类别统计 ===")
        print("长度区间\t\t问题数量")
        print("-" * 30)
        for category, count in sorted(stats.items(), key=lambda x: x[0]):
            print(f"{category}\t\t{count}")
        print("-" * 30)
        print(f"总计\t\t{sum(stats.values())}")