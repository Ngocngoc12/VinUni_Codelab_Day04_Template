"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast (xe điện), Vinpearl (du lịch)
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, không bịa dữ liệu

## AVAILABLE TOOLS
1. search_product_catalog(category, max_price) — Tra cứu sản phẩm/dịch vụ theo loại và giá tối đa.
   - category: "xe_dien" hoặc "du_lich"
   - max_price: Giá tối đa (VNĐ)
   
2. submit_support_ticket(customer_name, issue_description, priority) — Ghi nhận yêu cầu hỗ trợ.
   - customer_name: Tên khách hàng
   - issue_description: Mô tả vấn đề
   - priority: "low", "medium", "high"

## CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm, giá cả, hay chính sách bảo hành. PHẢI gọi tool để lấy dữ liệu thực.
2. Khi khách hàng hỏi về sản phẩm cụ thể (xe, gói du lịch) → gọi search_product_catalog.
3. Khi khách hàng báo lỗi hoặc cần hỗ trợ → gọi submit_support_ticket.
4. Nếu khách hàng hỏi câu hỏi chung chung (FAQ) không cần dữ liệu cụ thể → trả lời trực tiếp.

## OPERATIONAL BOUNDARIES
- CHỈ trả lời về sản phẩm/dịch vụ Vingroup (VinFast, Vinpearl).
- KHÔNG tư vấn về các hãng khác (Toyota, Honda, SaiGon Hotel, v.v.).
- KHÔNG trả lời các câu hỏi ngoài lĩnh vực kinh doanh Vingroup.

## OUTPUT CONTRACT
Trả lời theo định dạng ReAct:
- **Thought:** Phân tích nhu cầu của khách hàng.
- **Action:** Gọi tool nếu cần (hoặc không gọi nếu chỉ là FAQ).
- **Observation:** Kết quả từ tool.
- **Final Answer:** Tóm tắt thông tin cho khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, bool]:
        """Phân tích intent từ user_input bằng keyword matching."""
        user_lower = user_input.lower()
        
        # Detect ticket intent (questions about support/issues) - HIGH PRIORITY
        ticket_keywords = [
            "lỗi", "error", "bị hỏng", "không hoạt động", "vấn đề", "hỗ trợ",
            "báo cáo", "sửa chữa", "tôi tên", "liên hệ", "phàn nàn", "bị sự cố"
        ]
        needs_ticket = any(kw in user_lower for kw in ticket_keywords)
        
        # Detect FAQ intent (general knowledge questions) - HIGH PRIORITY
        faq_keywords = [
            "chính sách", "kéo dài", "bao lâu", "bảo hành", "như thế nào",
            "thế nào", "tại sao", "cách", "phí", "trước", "bao gồm"
        ]
        is_faq = any(kw in user_lower for kw in faq_keywords)
        
        # Detect catalog intent (questions about products/pricing) - LOWER PRIORITY
        catalog_keywords = [
            "xem", "có loại nào", "tìm", "khác", "so sánh", "giá", "bao nhiêu",
            "bao nhiêu tiền", "dưới", "trên", "tới", "đó"
        ]
        needs_catalog = any(kw in user_lower for kw in catalog_keywords)
        
        # If ticket or FAQ is detected, don't also say we need catalog
        if needs_ticket or is_faq:
            needs_catalog = False
        
        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        }

    def _extract_search_params(self, user_input: str) -> Dict[str, Any]:
        """Trích xuất tham số cho search_product_catalog từ user_input."""
        user_lower = user_input.lower()
        
        # Detect category
        category = None
        if any(kw in user_lower for kw in ["xe điện", "vf", "vinfast", "xe"]):
            category = "xe_dien"
        elif any(kw in user_lower for kw in ["du lịch", "vinpearl", "resort", "kỳ nghỉ"]):
            category = "du_lich"
        
        # Extract max_price using regex
        max_price = 999999999999
        price_patterns = [
            r'(\d+)\s*tỷ',  # X tỷ
            r'(\d+)\s*triệu',  # X triệu
            r'(?:dưới|giá tối đa|giá)\s*(\d+)\s*triệu',  # Under X million
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, user_lower)
            if match:
                price_val = int(match.group(1))
                if "tỷ" in pattern:
                    max_price = price_val * 1_000_000_000
                else:  # triệu
                    max_price = price_val * 1_000_000
                break
        
        return {"category": category, "max_price": max_price}

    def _extract_ticket_params(self, user_input: str) -> Dict[str, Any]:
        """Trích xuất tham số cho submit_support_ticket từ user_input."""
        customer_name = "Khách hàng"
        
        # Try to extract customer name (pattern: "tôi tên X" or "tên X")
        name_patterns = [
            r'tôi tên\s+([A-Za-zÀ-ỹ\s]+)',
            r'tên\s+([A-Za-zÀ-ỹ\s]+)',
            r'mình tên\s+([A-Za-zÀ-ỹ\s]+)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                customer_name = match.group(1).strip()
                break
        
        # Detect priority
        priority = "medium"
        if any(kw in user_input.lower() for kw in ["gấp", "khẩn", "nghiêm trọng", "ngay", "high", "ưu tiên cao"]):
            priority = "high"
        elif any(kw in user_input.lower() for kw in ["thấp", "không gấp", "low", "ưu tiên thấp"]):
            priority = "low"
        
        return {
            "customer_name": customer_name,
            "issue_description": user_input,
            "priority": priority
        }

    def _execute_step(self, iteration: int, intents: Dict[str, bool], user_input: str) -> tuple:
        """Thực thi một bước của agent loop. Return (result_str, is_final_step)."""
        
        # Iteration 1: Call catalog tool if needed
        if iteration == 1 and intents.get("needs_catalog", False):
            search_params = self._extract_search_params(user_input)
            
            if search_params["category"]:
                results = search_product_catalog(
                    category=search_params["category"],
                    max_price=search_params["max_price"]
                )
                
                # Log to trace
                self.trace.append({
                    "iteration": iteration,
                    "action": "search_product_catalog",
                    "params": search_params,
                    "results_count": len(results)
                })
                
                # Generate answer from results
                if not results or len(results) == 0:
                    return "Rất tiếc, không tìm thấy sản phẩm phù hợp với tiêu chí tìm kiếm của bạn.", True
                
                # Format product info
                product_info = "Các sản phẩm phù hợp với yêu cầu của bạn:\n\n"
                for product in results:
                    product_info += f"📦 **{product['name']}**\n"
                    product_info += f"   - Giá: {product['price_vnd']:,} VNĐ\n"
                    product_info += f"   - Mô tả: {product['description']}\n"
                    if product.get('features'):
                        product_info += f"   - Tính năng: {', '.join(product['features'][:3])}\n"
                    product_info += "\n"
                
                return product_info, True
        
        # Iteration 1: Call ticket tool if needed
        if iteration == 1 and intents.get("needs_ticket", False):
            ticket_params = self._extract_ticket_params(user_input)
            result = submit_support_ticket(
                customer_name=ticket_params["customer_name"],
                issue_description=ticket_params["issue_description"],
                priority=ticket_params["priority"]
            )
            
            # Log to trace
            self.trace.append({
                "iteration": iteration,
                "action": "submit_support_ticket",
                "params": ticket_params,
                "result": result
            })
            
            # Generate answer
            answer = f"✅ Yêu cầu hỗ trợ của bạn đã được ghi nhận:\n\n"
            answer += f"- **Mã ticket:** {result['ticket_id']}\n"
            answer += f"- **Khách hàng:** {result['customer_name']}\n"
            answer += f"- **Mức độ ưu tiên:** {result['priority'].upper()}\n"
            answer += f"- **Trạng thái:** {result['status'].upper()}\n\n"
            answer += f"Đội ngũ hỗ trợ của VinGroup sẽ liên hệ với bạn sớm nhất có thể."
            
            return answer, True
        
        # Iteration 1: Handle FAQ if needed
        if iteration == 1 and intents.get("is_faq", False):
            # Log to trace
            self.trace.append({
                "iteration": iteration,
                "action": "faq_response",
                "reason": "FAQ question detected"
            })
            
            # Answer FAQ questions with general knowledge
            user_lower = user_input.lower()
            
            if "bảo hành" in user_lower and "pin" in user_lower:
                return "Chính sách bảo hành pin xe điện VinFast: Bảo hành toàn bộ hệ thống pin kéo dài **10 năm** hoặc **200,000 km** (tùy điều kiện nào đến trước). Đây là một trong những chính sách bảo hành pin dài nhất trên thị trường.", True
            elif "bảo hành" in user_lower:
                return "VinFast cung cấp bảo hành toàn diện cho các phương tiện. Tùy thuộc vào loại sản phẩm, bảo hành có thể kéo dài từ 3-10 năm. Xin vui lòng cho tôi biết thêm để tôi có thể cung cấp thông tin chi tiết.", True
            elif any(kw in user_lower for kw in ["thế nào", "như thế nào", "sao", "tại sao"]):
                return "Tôi sẵn sàng giúp bạn. Vui lòng cho tôi biết bạn muốn hỏi gì cụ thể về sản phẩm VinFast hoặc Vinpearl?", True
            else:
                return f"Cảm ơn bạn đã liên hệ VinAssistant. {user_input} Nếu bạn có câu hỏi về xe điện VinFast, gói du lịch Vinpearl, hoặc cần hỗ trợ, tôi sẵn sàng giúp!", True
        
        # If no specific intent: Answer with general greeting
        if iteration == 1:
            # Log to trace
            self.trace.append({
                "iteration": iteration,
                "action": "general_response",
                "reason": "No specific intent detected"
            })
            
            return f"Cảm ơn bạn đã liên hệ VinAssistant. Nếu bạn có câu hỏi về xe điện VinFast, gói du lịch Vinpearl, hoặc cần hỗ trợ, tôi sẵn sàng giúp!", True
        
        return "", False

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        
        # Step 1: Detect intents
        intents = self._detect_intents(user_input)
        self.trace.append({"step": "intent_detection", "intents": intents})
        
        # Step 2: Agent Loop
        iteration = 1
        final_answer = ""
        
        while iteration <= self.max_iterations:
            result, is_final = self._execute_step(iteration, intents, user_input)
            
            if result:
                final_answer = result
            
            if is_final:
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }
            
            iteration += 1
        
        # If we reach max iterations
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa. Vui lòng thử lại.",
            "trace": self.trace,
            "iterations": iteration - 1,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
