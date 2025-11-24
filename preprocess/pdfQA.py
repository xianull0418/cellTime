#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdfplumber
import sys
import warnings
warnings.filterwarnings("ignore")

class DeepSeekR1PDFQA:
    def __init__(self, model_path: str, gpu_id: int = 0, max_context_length: int = 24000):
        """
        初始化DeepSeek-R1 PDF问答系统
        
        Args:
            model_path: 模型路径
            gpu_id: GPU设备ID
            max_context_length: 最大上下文长度（tokens）
        """
        self.model_path = model_path
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.max_context_length = max_context_length
        
        print(f"正在加载DeepSeek-R1模型从: {model_path}")
        print(f"使用设备: {self.device}")
        print(f"最大上下文长度: {max_context_length} tokens")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # 设置特殊token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            print("正在加载模型，这可能需要几分钟...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 确保模型在推理模式下
            self.model.eval()
            
            print("DeepSeek-R1模型加载完成!")
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("请确保已安装正确版本的transformers (>=4.37.0)")
            sys.exit(1)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从PDF文件中提取文本
        """
        print(f"正在从PDF提取文本: {pdf_path}")
        full_text = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        full_text += f"第{page_num}页:\n{text}\n\n"
                    print(f"已处理第 {page_num}/{total_pages} 页")
                    
        except Exception as e:
            print(f"提取PDF文本时出错: {e}")
            # 尝试使用PyPDF2作为备用
            try:
                full_text = self._extract_with_pypdf(pdf_path)
            except Exception as e2:
                print(f"PyPDF2提取也失败: {e2}")
                
        if not full_text.strip():
            print("警告: 未从PDF中提取到任何文本!")
            
        return full_text
    
    def _extract_with_pypdf(self, pdf_path: str) -> str:
        """使用PyPDF2提取文本"""
        try:
            import PyPDF2
            full_text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                for page_num in range(total_pages):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        full_text += f"第{page_num+1}页:\n{text}\n\n"
                    print(f"PyPDF2处理第 {page_num+1}/{total_pages} 页")
            return full_text
        except ImportError:
            print("请安装PyPDF2: pip install pypdf2")
            return ""
    
    def preprocess_context(self, text: str) -> str:
        """
        预处理上下文文本，按token数量截断
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        # 使用tokenizer计算token数量
        tokens = self.tokenizer.encode(text)
        token_count = len(tokens)
        
        print(f"原始文本token数量: {token_count}")
        
        # 如果token数量超过限制，按token截断而不是字符
        if token_count > self.max_context_length:
            print(f"文本过长，已截断至{self.max_context_length}个tokens")
            # 截断tokens并解码回文本
            truncated_tokens = tokens[:self.max_context_length]
            text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            print(f"截断后token数量: {len(self.tokenizer.encode(text))}")
        else:
            print(f"文本长度在限制内，无需截断")
            
        return text
    
    def create_deepseek_prompt(self, context: str, question: str) -> str:
        """
        创建DeepSeek-R1专用的提示模板
        """
        prompt = f"""请根据以下文档内容回答问题：

文档内容：
{context}

问题：{question}

请仔细阅读文档内容并给出准确的回答："""
        return prompt
    
    def generate_answer(self, question: str, context: str, max_length: int = 1024) -> str:
        """
        生成问题答案
        """
        prompt = self.create_deepseek_prompt(context, question)
        
        try:
            # 检查提示长度
            prompt_tokens = self.tokenizer.encode(prompt)
            print(f"提示token数量: {len(prompt_tokens)}")
            
            if len(prompt_tokens) > self.max_context_length:
                print(f"警告: 提示过长，可能影响模型性能")
            
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    early_stopping=True
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的答案部分
            answer = response[len(prompt):].strip()
            
            return answer
            
        except Exception as e:
            return f"生成答案时出错: {e}"
    
    def interactive_qa(self, pdf_path: str):
        """
        交互式问答循环
        """
        # 提取PDF文本
        context = self.extract_text_from_pdf(pdf_path)
        
        if not context.strip():
            print("没有可用的文本内容，无法进行问答。")
            return
        
        # 预处理上下文
        context = self.preprocess_context(context)
        
        print("\n" + "="*60)
        print("DeepSeek-R1 PDF问答系统已就绪！")
        print("输入 'quit' 或 'exit' 退出程序")
        print("输入 'show_context' 查看当前上下文")
        print("输入 'summary' 获取文档摘要")
        print("="*60)
        
        while True:
            try:
                question = input("\n请输入您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit']:
                    print("再见！")
                    break
                elif question.lower() == 'show_context':
                    # 显示更多上下文
                    print(f"\n当前上下文预览:\n{context[:1000]}...")
                    continue
                elif question.lower() == 'summary':
                    question = "请总结这篇文档的主要内容"
                
                if not question:
                    print("问题不能为空，请重新输入。")
                    continue
                
                print("\n正在生成答案，请稍候...")
                answer = self.generate_answer(question, context)
                
                print(f"\n答案: {answer}")
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断，再见！")
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="基于DeepSeek-R1-Distill-Qwen-7B的PDF问答系统")
    parser.add_argument("--model_path", type=str, required=True, help="DeepSeek-R1模型路径")
    parser.add_argument("--pdf_path", type=str, required=True, help="PDF文件路径")
    parser.add_argument("--gpu", type=int, default=0, help="GPU设备ID (默认: 0)")
    parser.add_argument("--max_tokens", type=int, default=24000, help="最大上下文token数量 (默认: 24000)")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.pdf_path):
        print(f"错误: PDF文件不存在: {args.pdf_path}")
        sys.exit(1)
    
    # 初始化问答系统
    qa_system = DeepSeekR1PDFQA(args.model_path, args.gpu, args.max_tokens)
    
    # 启动交互式问答
    qa_system.interactive_qa(args.pdf_path)

if __name__ == "__main__":
    main()
