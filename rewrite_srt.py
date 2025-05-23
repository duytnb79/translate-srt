import openai
import time
import os
import json
import uuid
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Initialize the OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Configuration ---
MODEL_PRICING = {
    # Prices per 1K tokens
    "gpt-4o-mini": {"prompt": 0.00000015, "completion": 0.0000006},
    "gpt-3.5-turbo": {"prompt": 0.0000005, "completion": 0.0000015},
    "gpt-4o": {"prompt": 0.000005, "completion": 0.000015},
    # Add other models and their pricing here if needed
}
rewrite_scripts_DIRECTORY = "rewrite_scripts" # Directory to store rewritten scripts
REWRITE_SUMMARIES_DIRECTORY = "rewrite_summaries" # Directory to store summaries for rewriting
DEFAULT_BATCH_SIZE = 10 # Default batch size for rewriting
DEFAULT_SUMMARIZATION_MODEL = "gpt-4o-mini" # Default model for summarization
DEFAULT_TITLE_DESCRIPTION_MODEL = "gpt-4o-mini" # Default model for titles & descriptions in rewrite context
NARRATIVE_SEGMENT_DURATION_SECONDS = 6 # Estimated duration for each auto-generated narrative segment
SUMMARY_SECTION_MARKERS = {
    "summary": "[CONTEXTUAL SUMMARY]",
    "titles": "[SUGGESTED TITLES]",
    "description": "[SUGGESTED DESCRIPTION]"
}

# --- Helper Functions ---

def srt_time_to_seconds(time_str: str) -> float:
    """Converts SRT time format (HH:MM:SS,ms) to seconds."""
    parts = time_str.split(',')
    hms = parts[0].split(':')
    return int(hms[0]) * 3600 + int(hms[1]) * 60 + int(hms[2]) + int(parts[1]) / 1000

def get_srt_total_duration(srt_blocks: list[str]) -> float:
    """Calculates the total duration of an SRT file from its blocks."""
    if not srt_blocks:
        return 0.0
    try:
        last_block = srt_blocks[-1]
        lines = last_block.split('\n')
        if len(lines) >= 2:
            timecode_line = lines[1]
            _, end_time_str = timecode_line.split(' --> ')
            return srt_time_to_seconds(end_time_str)
    except Exception as e:
        print(f"Lỗi khi tính toán tổng thời lượng SRT: {e}. Trả về 0.")
    return 0.0

def load_srt_blocks(input_path: str) -> list[str]:
    """Reads an SRT file and splits it into content blocks."""
    print(f"Đang đọc file: {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read().strip().split("\n\n")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {input_path}")
        return []
    except Exception as e:
        print(f"Lỗi khi đọc file {input_path}: {e}")
        return []

def generate_contextual_summary(srt_blocks: list[str], model: str) -> tuple[str, int, int]:
    """Generates a contextual summary of the entire SRT content to aid rewriting."""
    print(f"\nĐang tạo tóm tắt ngữ cảnh cho việc viết lại với model: {model}...")
    full_text_for_summary = []
    for block_text in srt_blocks:
        lines = block_text.split("\n")
        if len(lines) >= 3:
            full_text_for_summary.append(" ".join(lines[2:]))
    
    if not full_text_for_summary:
        print("Không có nội dung để tạo tóm tắt.")
        return "", 0, 0

    combined_text = "\n---".join(full_text_for_summary) # Removed extra newline

    summary_prompt = (
        "Bạn là một trợ lý phân tích nội dung chuyên sâu. Dựa vào toàn bộ nội dung kịch bản gốc được cung cấp dưới đây, "
        "hãy viết một bản tóm tắt bằng tiếng Việt (khoảng 200-350 từ). "
        "Bản tóm tắt này có hai mục đích chính: (1) Cung cấp thông tin nền về bối cảnh, cốt truyện tổng thể, các nhân vật quan trọng, mối quan hệ và động cơ của họ, cũng như giọng điệu và không khí chung của kịch bản. (2) QUAN TRỌNG HƠN: Xác định và liệt kê một cách rõ ràng CÁC Ý CHÍNH HOẶC CÁC SỰ KIỆN NÚT THẮT MANG TÍNH QUYẾT ĐỊNH của câu chuyện, tốt nhất là theo trình tự thời gian diễn ra. "
        "Những ý chính/sự kiện nút thắt này là những điểm cốt lõi mà một phiên bản kể lại tự sự SAU NÀY PHẢI ĐỀ CẬP ĐẾN để đảm bảo tính đầy đủ và trung thực với tinh thần của tác phẩm gốc. "
        "Phần liệt kê các ý chính này nên được trình bày một cách mạch lạc, dễ hiểu, có thể ở dạng gạch đầu dòng hoặc một đoạn văn riêng biệt nêu bật các điểm này một cách cô đọng. "
        "Ví dụ cách trình bày các ý chính (nếu dùng gạch đầu dòng): \n"
        "- Sự kiện A mở đầu câu chuyện, giới thiệu nhân vật X.\n"
        "- Xung đột chính nảy sinh khi Y xuất hiện.\n"
        "- Nhân vật X đưa ra quyết định quan trọng Z.\n"
        "- Cao trào của câu chuyện là sự kiện K.\n"
        "- Câu chuyện kết thúc với hậu quả M và bài học N.\n"
        "Mục tiêu cuối cùng là tạo ra một bản tóm tắt không chỉ mô tả mà còn cung cấp một 'dàn ý cốt truyện' vững chắc, chứa đựng những yếu tố không thể thiếu, để hỗ trợ việc tái tạo tự sự một cách sáng tạo nhưng vẫn bám sát các diễn biến và thông điệp quan trọng nhất của kịch bản gốc.\n\nNỘI DUNG KỊCH BẢN GỐC:\n"
        f"{combined_text}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý phân tích nội dung chuyên nghiệp, có khả năng nắm bắt các yếu tố quan trọng của kịch bản để hỗ trợ việc viết lại. Chỉ cung cấp tóm tắt bằng tiếng Việt."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.5,
            max_tokens=600
        )
        summary = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        print("Đã tạo xong tóm tắt ngữ cảnh cho việc viết lại.")
        return summary.strip(), prompt_tokens, completion_tokens
    except Exception as e:
        print(f"Lỗi khi tạo tóm tắt ngữ cảnh cho việc viết lại: {e}")
        return "", 0, 0

def parse_structured_summary_content(file_content: str) -> tuple[str, list[str], str]:
    """Parses the structured content from a summary file."""
    summary_text = ""
    titles_list = []
    description_text = ""
    current_section = None
    for line in file_content.splitlines():
        if line.strip() == SUMMARY_SECTION_MARKERS["summary"]:
            current_section = "summary"
            continue
        elif line.strip() == SUMMARY_SECTION_MARKERS["titles"]:
            current_section = "titles"
            continue
        elif line.strip() == SUMMARY_SECTION_MARKERS["description"]:
            current_section = "description"
            continue
        if current_section == "summary":
            summary_text += line + "\n"
        elif current_section == "titles":
            if line.strip():
                cleaned_line = line.strip()
                if len(cleaned_line) > 2 and cleaned_line[0].isdigit() and cleaned_line[1] == '.':
                    titles_list.append(cleaned_line[2:].strip())
                elif len(cleaned_line) > 1 and cleaned_line[0] == '-':
                    titles_list.append(cleaned_line[1:].strip())
                else:
                    titles_list.append(cleaned_line)
        elif current_section == "description":
            description_text += line + "\n"
    return summary_text.strip(), titles_list, description_text.strip()

def get_or_create_summary(
    srt_blocks: list[str], 
    summary_model: str, 
    title_model: str, 
    description_model: str, 
    summary_file_path: str
) -> tuple[str, list[str], str, int, int]:
    """Loads or generates summary, titles, and description for rewriting context."""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    if os.path.exists(summary_file_path):
        try:
            with open(summary_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            summary_text, titles_list, description_text = parse_structured_summary_content(file_content)
            if summary_text and titles_list and description_text:
                print(f"Đã tải tóm tắt, tiêu đề, mô tả (viết lại) từ: {summary_file_path}")
                return summary_text, titles_list, description_text, 0, 0
            else:
                print(f"File tóm tắt (viết lại) {summary_file_path} không đủ/lỗi. Sẽ tạo mới.")
        except Exception as e:
            print(f"Lỗi đọc/phân tích file tóm tắt (viết lại) {summary_file_path}: {e}. Sẽ tạo mới.")

    summary_text, s_p, s_c = generate_contextual_summary(srt_blocks, summary_model)
    total_prompt_tokens += s_p
    total_completion_tokens += s_c
    titles_list = []
    description_text = ""
    if summary_text:
        titles_list, t_p, t_c = generate_creative_titles_for_rewrite(summary_text, title_model)
        total_prompt_tokens += t_p
        total_completion_tokens += t_c
        description_text, d_p, d_c = generate_engaging_description_for_rewrite(summary_text, description_model)
        total_prompt_tokens += d_p
        total_completion_tokens += d_c
    else:
        print("Không tạo được tóm tắt (viết lại), bỏ qua tạo tiêu đề/mô tả.")
    if summary_text or titles_list or description_text:
        try:
            os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
            with open(summary_file_path, "w", encoding="utf-8") as f:
                f.write(f"{SUMMARY_SECTION_MARKERS['summary']}\n{summary_text}\n\n")
                f.write(f"{SUMMARY_SECTION_MARKERS['titles']}\n")
                for i, title in enumerate(titles_list):
                    f.write(f"{i+1}. {title}\n")
                f.write("\n")
                f.write(f"{SUMMARY_SECTION_MARKERS['description']}\n{description_text}\n")
            print(f"Đã lưu tóm tắt, tiêu đề, mô tả (viết lại) vào: {summary_file_path}")
        except Exception as e:
            print(f"Lỗi lưu file tóm tắt tổng hợp (viết lại) {summary_file_path}: {e}")
    return summary_text, titles_list, description_text, total_prompt_tokens, total_completion_tokens

def generate_creative_titles_for_rewrite(context_summary: str, model: str) -> tuple[list[str], int, int]:
    """Generates creative titles for rewriting context, optimized for TikTok SEO."""
    print(f"\nĐang tạo tiêu đề sáng tạo (TikTok SEO, viết lại) với model: {model}...")
    if not context_summary: return [], 0, 0
    prompt = (
        "Bạn là một chuyên gia sáng tạo nội dung và tối ưu SEO cho TikTok. "
        "Dựa vào tóm tắt kịch bản được cung cấp, hãy đề xuất 3-5 tiêu đề video tiếng Việt cực kỳ thu hút cho TikTok. "
        "Các tiêu đề này cần:\n"
        "1. Ngắn gọn (tối đa 70-100 ký tự nếu có thể, hoặc 1-2 dòng ngắn trên màn hình TikTok).\n"
        "2. Tạoフック mạnh (strong hook): Gây ấn tượng ngay từ những từ đầu tiên.\n"
        "3. Kích thích tò mò cao độ: Khiến người xem phải dừng lại và xem video ngay lập tức.\n"
        "4. Có thể gợi ý từ khóa hoặc chủ đề đang thịnh hành (nếu có thể suy luận chung từ nội dung).\n"
        "5. Phản ánh nội dung cốt lõi hoặc điểm đặc sắc nhất của kịch bản (đã được tóm tắt) theo một cách gây sốc hoặc bất ngờ.\n\n"
        f"TÓM TẮT KỊCH BẢN: '''{context_summary}'''\n\n"
        "Ví dụ phong cách (không phải nội dung): 'Sự thật đằng sau... [Gây Sốc]', 'Bạn sẽ KHÔNG TIN NỔI khi thấy...', 'POV: Lần đầu tôi...'\n"
        "Chỉ cung cấp 3-5 tiêu đề, mỗi tiêu đề một dòng."
    )
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": "Chuyên gia sáng tạo tiêu đề video TikTok tiếng Việt, ưu tiên sự ngắn gọn, gây sốc và tò mò."}, {"role": "user", "content": prompt}],
            temperature=0.8, # Slightly higher for more viral-style creativity
            max_tokens=250 
        )
        titles = [t.strip() for t in response.choices[0].message.content.split("\n") if t.strip()]
        return titles, response.usage.prompt_tokens if response.usage else 0, response.usage.completion_tokens if response.usage else 0
    except Exception as e:
        print(f"Lỗi tạo tiêu đề (viết lại, TikTok): {e}"); return [], 0, 0

def generate_engaging_description_for_rewrite(context_summary: str, model: str) -> tuple[str, int, int]:
    """Generates an engaging description for rewriting context, optimized for TikTok SEO."""
    print(f"Đang tạo mô tả hấp dẫn (TikTok SEO, viết lại) với model: {model}...")
    if not context_summary: return "", 0, 0
    prompt = (
        "Bạn là một chuyên gia sáng tạo nội dung và copywriter cho TikTok. "
        "Dựa vào tóm tắt kịch bản được cung cấp, hãy viết một mô tả (caption) video TikTok bằng tiếng Việt siêu ngắn (tối đa 1-3 câu, khoảng 100-220 ký tự). "
        "Mô tả này cần:\n"
        "1. Có MÓC CÂU CỰC MẠNH ở ngay đầu.\n"
        "2. Tạo sự tò mò hoặc đặt câu hỏi để khuyến khích tương tác (bình luận, xem hết video).\n"
        "3. Liên quan trực tiếp đến điểm hấp dẫn nhất của video (dựa trên tóm tắt).\n"
        "4. Có thể bao gồm 2-3 hashtag gợi ý chung chung liên quan đến chủ đề (ví dụ: #learnontiktok #bian #khampha #kechuyen). Tránh hashtag quá cụ thể hoặc trend nhất thời mà AI có thể không biết.\n\n"
        f"TÓM TẮT KỊCH BẢN: '''{context_summary}'''\n\n"
        "Ví dụ phong cách caption (không phải nội dung): 'Không thể tin được chuyện gì đã xảy ra 😱 Xem ngay để biết! #shock #batngo', 'Bạn nghĩ sao về điều này? 🤔 Comment ngay! #learnontiktok #xuhuong'\n"
        "Chỉ cung cấp nội dung mô tả (caption) và các hashtag đề xuất."
    )
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": "Chuyên gia viết caption TikTok tiếng Việt siêu thu hút, ngắn gọn, kèm hashtag liên quan."}, {"role": "user", "content": prompt}],
            temperature=0.75, # Maintain creativity for description
            max_tokens=200 # Descriptions are short
        )
        return response.choices[0].message.content.strip(), response.usage.prompt_tokens if response.usage else 0, response.usage.completion_tokens if response.usage else 0
    except Exception as e:
        print(f"Lỗi tạo mô tả (viết lại, TikTok): {e}"); return "", 0, 0

def generate_new_narrative_segments(context_summary: str, original_srt_full_text: str, target_duration_seconds: float, model: str, narrative_goal_prompt_template: str) -> tuple[list[str], int, int]:
    """Generates new narrative segments based on summary, full original text, target duration, and a goal prompt template."""
    print(f"\nĐang tạo các phân đoạn tự sự mới với model: {model}...")
    
    if not context_summary and not original_srt_full_text:
        print("Lỗi: Cần tóm tắt ngữ cảnh hoặc toàn bộ kịch bản gốc để tạo tự sự mới.")
        return [], 0, 0

    # Construct the dynamic part of the prompt regarding duration
    duration_guidance = f"Tổng thời lượng của kịch bản gốc là khoảng {target_duration_seconds:.0f} giây. Kịch bản tự sự mới bạn tạo ra nên có tổng thời lượng tương đương hoặc dài hơn một chút. Hãy cân nhắc điều này khi quyết định số lượng và độ dài của các phân đoạn."
    if target_duration_seconds == 0:
        duration_guidance = "Không có thông tin thời lượng gốc, hãy tập trung vào việc tạo đủ số lượng phân đoạn để bao phủ nội dung một cách hợp lý."

    full_prompt = (
        f"{narrative_goal_prompt_template.format(duration_guidance=duration_guidance)}\n\n"
        "TÓM TẮT KỊCH BẢN ĐỂ THAM KHẢO (cho bối cảnh và giọng điệu tổng thể):\n"
        f"'''{context_summary}'''\n\n"
        "KỊCH BẢN GỐC ĐẦY ĐỦ (để đảm bảo bám sát các sự kiện và thông tin chi tiết):\n"
        f"'''{original_srt_full_text}'''\n\n"
        "LƯU Ý QUAN TRỌNG: Chỉ trả về các đoạn văn bản, mỗi đoạn cách nhau bởi dấu phân tách '||NEW_SEGMENT||'. Không thêm bất kỳ giải thích, tiêu đề, hay định dạng nào khác."
    )

    prompt_tokens = 0
    completion_tokens = 0
    segments = []

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là một nhà văn sáng tạo, chuyên gia xây dựng kịch bản tự sự hấp dẫn bằng tiếng Việt. Nhiệm vụ của bạn là tạo ra các đoạn văn bản tường thuật dựa trên tóm tắt và hướng dẫn được cung cấp. Luôn trả lời bằng các đoạn văn bản phân tách bởi '||NEW_SEGMENT||'."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.75, # Slightly higher for more creativity
            max_tokens=3000  # Allow for a decent length narrative
        )

        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
        
        raw_response_content = response.choices[0].message.content
        if raw_response_content:
            segments = [seg.strip() for seg in raw_response_content.split("||NEW_SEGMENT||") if seg.strip()]
            if not segments:
                 print(f"Cảnh báo: API trả về nội dung nhưng không tìm thấy phân đoạn hợp lệ sau khi tách bằng '||NEW_SEGMENT||'. Nội dung thô: {raw_response_content[:200]}...")
        else:
            print("Lỗi: API trả về nội dung trống cho việc tạo tự sự mới.")

    except Exception as e:
        print(f"Lỗi khi tạo phân đoạn tự sự mới: {e}")
        # segments will remain empty

    if segments:
        print(f"Đã tạo được {len(segments)} phân đoạn tự sự mới.")
    else:
        print("Không tạo được phân đoạn tự sự nào.")
        
    return segments, prompt_tokens, completion_tokens

def rewrite_single_block_content(original_text: str, model: str, rewrite_instruction: str) -> tuple[str, int, int]:
    """
    Rewrites the text content of a single SRT block using OpenAI API.
    Returns the rewritten text, prompt tokens, and completion tokens.
    """
    if not original_text.strip():
        return original_text, 0, 0 # Return original if empty or only whitespace

    system_message_content = (
        "Bạn là một trợ lý viết lại nội dung chuyên nghiệp. "
        "Nhiệm vụ của bạn là viết lại văn bản được cung cấp theo hướng dẫn cụ thể, "
        "giữ nguyên ý nghĩa cốt lõi nhưng thay đổi văn phong hoặc cấu trúc theo yêu cầu."
    )
    
    user_prompt = (
        f"{rewrite_instruction}\n\n"
        "Văn bản gốc:\n"
        f"'''{original_text}'''\n\n"
        "Chỉ trả về phần văn bản đã được viết lại, không thêm bất kỳ lời giải thích hay bình luận nào khác."
    )

    prompt_tokens = 0
    completion_tokens = 0
    rewritten_text = f"[LỖI VIẾT LẠI - {original_text[:30]}...]" # Default error message

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7, # Adjust temperature as needed for creativity vs. fidelity
            max_tokens=1024  # Max tokens for the rewritten output of a single block
        )

        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
        
        rewritten_text_from_api = response.choices[0].message.content
        if rewritten_text_from_api:
            rewritten_text = rewritten_text_from_api.strip()
        else:
            print(f"Lỗi: API trả về nội dung trống cho khối: {original_text[:50]}...")
            # Keep default error message for this case

    except Exception as e:
        print(f"Lỗi khi viết lại khối '{original_text[:50]}...': {e}")
        # Keeps the default error message with original text snippet

    return rewritten_text, prompt_tokens, completion_tokens

def rewrite_batch_content(original_texts_in_batch: list[tuple[str, str]], model: str, rewrite_instruction: str, context_summary: str) -> tuple[list[str], int, int]:
    """
    Rewrites a batch of SRT text blocks using OpenAI API with a single call, using context summary.
    original_texts_in_batch is a list of tuples (item_id, text_to_rewrite).
    Returns a list of rewritten texts in the same order, prompt tokens, and completion tokens.
    """
    if not original_texts_in_batch:
        return [], 0, 0

    prompt_text_parts = []
    for item_id, text_content in original_texts_in_batch:
        escaped_text_content = text_content.replace("'''", "\'\'\'")
        prompt_text_parts.append(f'{item_id}: """{escaped_text_content}"""')
    
    prompt_text_for_api = "\n".join(prompt_text_parts)

    system_message_content = (
        "Bạn là một trợ lý viết lại nội dung chuyên nghiệp. "
        "Nhiệm vụ của bạn là viết lại các đoạn văn bản được cung cấp theo hướng dẫn cụ thể, "
        "giữ nguyên ý nghĩa cốt lõi nhưng thay đổi văn phong hoặc cấu trúc theo yêu cầu. "
        "Sử dụng tóm tắt kịch bản được cung cấp để đảm bảo tính nhất quán và phù hợp với ngữ cảnh. "
        "Xử lý từng mục văn bản một cách độc lập."
    )

    user_prompt = (
        f"{rewrite_instruction}\n\n"
        "Hãy viết lại các đoạn văn bản được đánh số ID dưới đây. "
        "Tham khảo TÓM TẮT KỊCH BẢN sau để hiểu rõ hơn về bối cảnh:\n\n"
        f"TÓM TẮT KỊCH BẢN:\n'''{context_summary}'''\n\n"
        "Cung cấp các phiên bản đã viết lại trong một đối tượng JSON hợp lệ, "
        "trong đó mỗi khóa là ID của mục (ví dụ: 'item_0') và giá trị là văn bản đã được viết lại tương ứng.\n\n"
        "Văn bản gốc cần viết lại:\n"
        f"{prompt_text_for_api}\n\n"
        "Ví dụ định dạng JSON đầu ra: {\"item_0\": \"văn bản viết lại cho item_0\", \"item_1\": \"văn bản viết lại cho item_1\", ...}"
    )

    prompt_tokens = 0
    completion_tokens = 0
    rewritten_texts_for_batch = [f"[LỖI VIẾT LẠI BATCH - {text[:30]}...]" for _, text in original_texts_in_batch] 

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=3500, 
            response_format={"type": "json_object"}
        )

        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
        
        rewritten_content_json_str = response.choices[0].message.content
        if not rewritten_content_json_str:
            print(f"Lỗi: API trả về nội dung trống cho batch.")
            return rewritten_texts_for_batch, prompt_tokens, completion_tokens

        rewritten_content_json = json.loads(rewritten_content_json_str)

        for i, (item_id, original_text) in enumerate(original_texts_in_batch):
            rewritten_text = rewritten_content_json.get(item_id)
            if rewritten_text:
                rewritten_texts_for_batch[i] = rewritten_text.strip()
            else:
                print(f"Cảnh báo: Không tìm thấy mục '{item_id}' trong phản hồi JSON cho batch. Giữ nguyên văn bản gốc hoặc lỗi.")
                rewritten_texts_for_batch[i] = f"[LỖI - KHÔNG TÌM THẤY {item_id} TRONG PHẢN HỒI BATCH]"

    except json.JSONDecodeError as e:
        print(f"Lỗi khi giải mã JSON từ API cho batch: {e}. Dữ liệu nhận được: {rewritten_content_json_str if 'rewritten_content_json_str' in locals() else 'Không có dữ liệu'}")
    except Exception as e:
        print(f"Lỗi khi viết lại batch: {e}")
            
    return rewritten_texts_for_batch, prompt_tokens, completion_tokens

def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculates the estimated cost based on token usage and model pricing."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        print(f"CẢNH BÁO: Không tìm thấy thông tin giá cho model '{model}'. Chi phí sẽ không được tính.")
        return 0.0

    prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1000) * pricing["completion"]
    total_cost = prompt_cost + completion_cost
    return total_cost

def save_results_to_file(output_path: str, results: list[str]):
    """Saves the rewritten SRT blocks to the specified output file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(results)) 
    except Exception as e:
        print(f"Lỗi khi lưu file {output_path}: {e}")

# --- Main Rewriting Function ---

def rewrite_srt_script_main(
    input_path: str, 
    output_path_placeholder: str, # Placeholder, actual output path determined by mode
    model: str, 
    rewrite_instruction_block: str, # Instruction for block-by-block
    batch_size: int, 
    summary_model: str, 
    title_model: str, # New parameter
    description_model: str, # New parameter
    summary_file_path: str,
    rewrite_mode: str,
    block_rewrite_style: str # "formal" or "creative_text"
):
    """Main function to rewrite/regenerate an SRT file's content based on the selected mode."""
    srt_blocks_full = load_srt_blocks(input_path)
    if not srt_blocks_full and rewrite_mode == "block_by_block_rewrite": # Only critical if blocks are needed
        print("Không có nội dung SRT để xử lý cho chế độ viết lại theo khối. Kết thúc.")
        return
    if not srt_blocks_full and rewrite_mode == "narrative_regeneration" and not os.path.exists(summary_file_path):
        print("Không có nội dung SRT để tạo tóm tắt cho chế độ tạo tự sự mới và không có tóm tắt sẵn. Kết thúc.")
        return

    grand_total_prompt_tokens = 0
    grand_total_completion_tokens = 0
    total_estimated_cost = 0.0

    # --- Step 1: Contextual Summary, Titles, Description (Common for both modes) ---
    print("\n--- Bước 1: Tạo/Tải tóm tắt, tiêu đề & mô tả (cho viết lại) ---")
    context_summary, generated_titles, generated_description, s_t_d_p_tokens, s_t_d_c_tokens = get_or_create_summary(
        srt_blocks_full if srt_blocks_full else [], 
        summary_model, 
        title_model, 
        description_model, 
        summary_file_path
    )
    grand_total_prompt_tokens += s_t_d_p_tokens
    grand_total_completion_tokens += s_t_d_c_tokens

    # Combined cost for summary, titles, description
    s_t_d_cost = 0.0
    if s_t_d_p_tokens > 0 or s_t_d_c_tokens > 0:
        # Approximating cost with summary_model, or title_model if different pricing tier
        s_t_d_cost = calculate_cost(s_t_d_p_tokens, s_t_d_c_tokens, title_model) 
        print(f"Chi phí tóm tắt, tiêu đề & mô tả (ước tính với model: {title_model}): ${s_t_d_cost:.6f}")
    total_estimated_cost += s_t_d_cost

    if not context_summary:
        print("CẢNH BÁO: Không thể tạo hoặc tải tóm tắt. Chức năng có thể bị hạn chế.")
        if rewrite_mode == "narrative_regeneration":
            print("Không thể tiếp tục chế độ tạo tự sự mới mà không có tóm tắt. Kết thúc.")
            return
    else:
        print("\n--- Thông tin bổ trợ (viết lại) đã tạo/tải ---")
        print(f"Tóm tắt ngữ cảnh:\n{context_summary}")
        if generated_titles:
            print("\nCác đề xuất tiêu đề (cho viết lại):")
            for i, title in enumerate(generated_titles):
                print(f"  {i+1}. {title}")
        else:
            print("\nKhông có tiêu đề nào được tạo/tải (cho viết lại).")
        if generated_description:
            print(f"\nMô tả đề xuất (cho viết lại):\n{generated_description}")
        else:
            print("\nKhông có mô tả nào được tạo/tải (cho viết lại).")
    
    # Determine actual output path based on mode and original output_path_placeholder structure
    output_directory = os.path.dirname(output_path_placeholder)
    base_name_with_identifier = os.path.basename(output_path_placeholder).replace(".srt", "") #e.g. 0522_placeholder_timestamp
    original_base_name = base_name_with_identifier.split("_")[0] #e.g. 0522
    # The identifier is now a timestamp, not a UUID
    run_identifier = base_name_with_identifier.split("_")[-1] #e.g. timestamp_str 

    actual_output_path = ""

    # --- Mode Selection ---
    if rewrite_mode == "block_by_block_rewrite":
        print(f"\n--- Chế độ: Viết lại theo khối (Block-by-Block Rewrite) - Kiểu: {block_rewrite_style} ---")
        if not srt_blocks_full:
            print("Không có khối SRT nào để viết lại. Kết thúc.")
            return
        
        actual_output_filename = f"{original_base_name}_block_rewrite_{block_rewrite_style}_{run_identifier}.srt"
        actual_output_path = os.path.join(output_directory, actual_output_filename)

        print(f"Sử dụng model viết lại: {model}, kích thước batch: {batch_size}")
        print(f"Hướng dẫn viết lại: {rewrite_instruction_block}")
        if context_summary:
            print("Sử dụng tóm tắt ngữ cảnh để hỗ trợ viết lại.")
        else:
            print("CẢNH BÁO: Viết lại mà không có tóm tắt ngữ cảnh bổ sung.")

        rewrite_section_p_tokens = 0
        rewrite_section_c_tokens = 0
        
        structured_srt_entries = []
        for block_idx, block_text_original in enumerate(srt_blocks_full):
            lines = block_text_original.split("\n")
            if len(lines) < 3:
                structured_srt_entries.append({'type': 'malformed', 'content': block_text_original})
            else:
                index, timecode = lines[0], lines[1]
                text_to_rewrite = "\n".join(lines[2:])
                structured_srt_entries.append({
                    'type': 'valid',
                    'id': f"item_{block_idx}",
                    'index': index,
                    'timecode': timecode,
                    'original_text': text_to_rewrite
                })

        valid_entries_to_process = [entry for entry in structured_srt_entries if entry['type'] == 'valid']
        batches = [valid_entries_to_process[i:i + batch_size] for i in range(0, len(valid_entries_to_process), batch_size)]
        processed_text_map = {}

        for batch_of_entries in tqdm(batches, desc="Đang viết lại các batch", unit="batch"):
            texts_for_api_batch = [(entry['id'], entry['original_text']) for entry in batch_of_entries]
            if not texts_for_api_batch: continue

            rewritten_texts_for_current_batch, p_tokens, c_tokens = rewrite_batch_content(
                texts_for_api_batch, model, rewrite_instruction_block, context_summary if context_summary else ""
            )
            
            rewrite_section_p_tokens += p_tokens
            rewrite_section_c_tokens += c_tokens

            for entry, rewritten_text in zip(batch_of_entries, rewritten_texts_for_current_batch):
                processed_text_map[entry['id']] = rewritten_text
            
            current_all_rewritten_srt_blocks = []
            for entry in structured_srt_entries:
                if entry['type'] == 'malformed':
                    current_all_rewritten_srt_blocks.append(entry['content'])
                elif entry['type'] == 'valid':
                    rewritten_content = processed_text_map.get(entry['id'], entry['original_text']) 
                    current_all_rewritten_srt_blocks.append(f"{entry['index']}\n{entry['timecode']}\n{rewritten_content}")
            save_results_to_file(actual_output_path, current_all_rewritten_srt_blocks)
            time.sleep(1.0)
        
        grand_total_prompt_tokens += rewrite_section_p_tokens
        grand_total_completion_tokens += rewrite_section_c_tokens

        rewrite_cost = calculate_cost(rewrite_section_p_tokens, rewrite_section_c_tokens, model)
        print(f"Chi phí viết lại (model: {model}): ${rewrite_cost:.6f}")
        total_estimated_cost += rewrite_cost
        print(f"Kịch bản viết lại theo khối đã được lưu tại: {actual_output_path}")

    elif rewrite_mode == "narrative_regeneration":
        print(f"\n--- Chế độ: Tạo tự sự mới (Narrative Re-generation) ---")
        if not context_summary:
            print("Không có tóm tắt, không thể tạo tự sự mới. Kết thúc.")
            return # Already checked, but as a safeguard

        actual_output_filename = f"{original_base_name}_narrative_regen_{run_identifier}.srt"
        actual_output_path = os.path.join(output_directory, actual_output_filename)
        
        print(f"Sử dụng model tạo tự sự: {model}")
        narrative_goal_prompt_template = (
            "Bạn LÀ Bà Diệu An, một người kể chuyện lớn tuổi, đức độ và am hiểu sự đời. Hãy hoàn toàn nhập vai Bà Diệu An. "
            "QUAN TRỌNG VỀ XƯNG HÔ: Khi kể chuyện, hãy LUÔN LUÔN xưng là 'bà' (ví dụ: 'bà kể cho các con nghe', 'ngày đó bà còn trẻ...') và gọi người nghe (khán giả) là 'các con' (ví dụ: 'các con có biết không?', 'chuyện là thế này nè các con...'). "
            "Hãy kể lại câu chuyện này như thể chính bà đang thủ thỉ, tâm tình, chia sẻ những ký ức sâu sắc của mình trực tiếp với các con. Lời kể của bà phải thật tự nhiên, chân thành, thể hiện đúng vai vế và tình cảm của một người bà với các con cháu. Lời kể của bà chính là lời của Bà Diệu An, không phải là một sự tường thuật lại từ một người thứ ba hay một diễn viên đóng vai. "
            
            "LƯU Ý ĐẶC BIỆT VỀ NHÂN VẬT TRONG KỊCH BẢN GỐC: Trong kịch bản gốc, nếu có nhắc đến tên 'Diệu Huyền', các con hãy hiểu rằng đó chính là bà (Bà Diệu An) khi còn trẻ hoặc trong một bối cảnh khác của câu chuyện. Khi kể lại, bà sẽ luôn xưng là 'bà' khi nói về mình trong vai Diệu Huyền/Bà Diệu An, và kể từ góc nhìn của bà, như đang kể cho các con nghe về quá khứ của mình. "
            
            "Kịch bản tự sự mới này phải bám sát chặt chẽ các sự kiện, thông tin và trình tự của KỊCH BẢN GỐC ĐẦY ĐỦ. "
            "Sử dụng TÓM TẮT KỊCH BẢN để nắm bắt giọng điệu và bối cảnh chung, nhưng toàn bộ lời kể và cách xưng hô phải là của Bà Diệu An nói với các con. "
            "{duration_guidance} " 
            "Chia kịch bản thành nhiều đoạn văn ngắn (khoảng 2-4 câu mỗi đoạn), mỗi đoạn văn sẽ tương ứng với một khối phụ đề. Lời văn cần tự nhiên, mang đậm tính kể chuyện của Bà Diệu An (xưng 'bà', gọi 'các con'), có thể có những lời bình luận ý nhị hoặc chiêm nghiệm của chính bà. Khi mô tả các nhân vật khác hoặc đưa ra những lời bình/chiêm nghiệm này, hãy đảm bảo giọng điệu, cách nhìn, và ngôn từ phải hoàn toàn là của Bà Diệu An – một người phụ nữ lớn tuổi, từng trải, am hiểu sự đời, đang kể chuyện cho các con nghe. Toàn bộ lời kể, bao gồm cả việc mô tả nhân vật và sự kiện, đều phải nhất quán với vai kể này. "
            "Tập trung vào việc kể một câu chuyện mạch lạc, thu hút sự chú ý của các con, có thể bao gồm các yếu tố bất ngờ hoặc cảm xúc, nhưng không được thay đổi các sự thật cốt lõi từ kịch bản gốc (ngoại trừ việc thống nhất tên gọi và cách xưng hô như đã nêu trên). "
            "Mục tiêu là tạo ra một trải nghiệm xem mới mẻ, với giọng kể đặc trưng, thân mật và trực tiếp của Bà Diệu An (bà xưng 'bà', gọi 'các con'), có thể dùng cho hình ảnh gốc hoặc gợi ý hình ảnh mới, đồng thời vẫn truyền tải trung thực nội dung gốc. "
            "Vui lòng trả về một danh sách các đoạn văn bản (phân tách bằng một dấu xuống dòng đặc biệt như '||NEW_SEGMENT||' giữa các đoạn) mà không có bất kỳ định dạng JSON hay đánh số nào."
        )
        
        original_srt_full_text = "\n\n".join(srt_blocks_full) 
        original_duration_seconds = get_srt_total_duration(srt_blocks_full)
        print(f"Tổng thời lượng kịch bản gốc ước tính: {original_duration_seconds:.2f} giây.")

        print("Đang tạo tự sự mới...")
        segments, nr_p_tokens, nr_c_tokens = generate_new_narrative_segments(
            context_summary if context_summary else "", 
            original_srt_full_text,
            original_duration_seconds,
            model, 
            narrative_goal_prompt_template
        )

        if segments:
            generated_srt_content = []
            start_time_ms = 0
            
            # Distribute original duration among segments
            num_segments = len(segments)
            if num_segments > 0 and original_duration_seconds > 0:
                avg_duration_per_segment_ms = (original_duration_seconds * 1000) / num_segments
            else:
                avg_duration_per_segment_ms = NARRATIVE_SEGMENT_DURATION_SECONDS * 1000 # Fallback
            
            # Ensure minimum duration to avoid overly short segments if original is very short or many segments generated
            min_segment_duration_ms = 3000 # 3 seconds minimum
            segment_duration_ms = max(avg_duration_per_segment_ms, min_segment_duration_ms)

            print(f"Sẽ tạo {num_segments} khối phụ đề mới, mỗi khối khoảng {segment_duration_ms/1000:.2f} giây.")
            for i, seg_text in enumerate(tqdm(segments, desc="Đang tạo các khối SRT từ tự sự mới", unit="khối")):
                end_time_ms = start_time_ms + segment_duration_ms
                
                # Ensure ms components are integers for formatting
                current_start_ms_int = int(start_time_ms % 1000)
                current_end_ms_int = int(end_time_ms % 1000)

                start_hms = time.strftime('%H:%M:%S', time.gmtime(start_time_ms // 1000))
                end_hms = time.strftime('%H:%M:%S', time.gmtime(end_time_ms // 1000))
                
                timecode = f"{start_hms},{current_start_ms_int:03d} --> {end_hms},{current_end_ms_int:03d}"
                generated_srt_content.append(f"{i+1}\n{timecode}\n{seg_text}")
                start_time_ms = end_time_ms + 200 # Small gap between subtitles
            save_results_to_file(actual_output_path, generated_srt_content)
            print(f"Kịch bản tự sự mới (với thời gian tự động cơ bản) đã được lưu tại: {actual_output_path}")

            grand_total_prompt_tokens += nr_p_tokens
            grand_total_completion_tokens += nr_c_tokens
            narrative_regen_cost = calculate_cost(nr_p_tokens, nr_c_tokens, model)
            print(f"Chi phí tạo tự sự mới (model: {model}): ${narrative_regen_cost:.6f}")
            total_estimated_cost += narrative_regen_cost
        else:
            print("Không tạo được phân đoạn tự sự mới.")

    else:
        print(f"LỖI: Chế độ viết lại '{rewrite_mode}' không hợp lệ.")
        return

    print("\n--- Hoàn Thành Xử Lý ---")
    # print(f"Kết quả đã được lưu tại các đường dẫn tương ứng với chế độ đã chọn.") # More specific messages are printed within modes
    print(f"Tổng số prompt tokens (bao gồm tóm tắt, và viết lại/tạo tự sự): {grand_total_prompt_tokens}")
    print(f"Tổng số completion tokens (bao gồm tóm tắt, và viết lại/tạo tự sự): {grand_total_completion_tokens}")
    print(f"Tổng ước tính chi phí cho phiên làm việc: ${total_estimated_cost:.6f}")


# --- Script Execution ---

if __name__ == "__main__":
    INPUT_FILE_PATH = "data/0523.srt"  
    REWRITE_MODEL_NAME = "gpt-4o-mini" # Used for both block rewrite and narrative regen for now
    SUMMARIZATION_MODEL_NAME = DEFAULT_SUMMARIZATION_MODEL
    TITLE_DESCRIPTION_MODEL_NAME = DEFAULT_TITLE_DESCRIPTION_MODEL # New constant for rewrite context
    SRT_BATCH_SIZE = DEFAULT_BATCH_SIZE # Only for block_by_block_rewrite mode
    
    # --- CHOOSE OPERATIONAL MODE ---
    REWRITE_MODE = "block_by_block_rewrite"  # Options: "narrative_regeneration", "block_by_block_rewrite"
    # -------------------------------

    # --- CHOOSE REWRITE STYLE (Only for REWRITE_MODE = "block_by_block_rewrite") ---
    BLOCK_REWRITE_STYLE = "creative_text" # Options: "formal", "creative_text"
    # ----------------------------------------------------------------------------

    REWRITE_INSTRUCTION_FORMAL = (
        "Hãy viết lại đoạn văn bản sau đây với văn phong trang trọng hơn, "
        "thích hợp cho một giọng đọc phim tài liệu. Giữ nguyên ý nghĩa cốt lõi của văn bản. "
        "Không thay đổi tên riêng hoặc các thuật ngữ kỹ thuật trừ khi được yêu cầu."
    )

    REWRITE_INSTRUCTION_CREATIVE_TEXT = (
        "Hãy viết lại đoạn văn bản sau đây để trở nên hấp dẫn, thu hút sự tò mò của khán giả đại chúng. "
        "Bạn có thể sử dụng các kỹ thuật kể chuyện, thêm yếu tố gây cấn, hoặc đơn giản hóa các ý tưởng phức tạp để nội dung dễ tiếp cận và dễ lan tỏa hơn. "
        "Mục tiêu là tối đa hóa sự duy trì và hứng thú của người xem, ngay cả khi điều đó có nghĩa là thay đổi một chút cách diễn đạt trực tiếp của kịch bản gốc, nhưng vẫn phải giữ được các sự kiện hoặc thông tin cốt lõi. "
        "Tham khảo tóm tắt ngữ cảnh để đảm bảo sự sáng tạo phù hợp với bức tranh toàn cảnh của câu chuyện."
    )

    SELECTED_BLOCK_REWRITE_INSTRUCTION = ""
    output_filename_suffix_placeholder = "placeholder" # This will be refined by the main function

    if BLOCK_REWRITE_STYLE == "formal":
        SELECTED_BLOCK_REWRITE_INSTRUCTION = REWRITE_INSTRUCTION_FORMAL
    elif BLOCK_REWRITE_STYLE == "creative_text":
        SELECTED_BLOCK_REWRITE_INSTRUCTION = REWRITE_INSTRUCTION_CREATIVE_TEXT
    # No exit here if style is wrong, main function can use default or handle if mode is block_by_block

    # Generate a timestamp string for filenames
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    if not os.path.exists(rewrite_scripts_DIRECTORY):
        os.makedirs(rewrite_scripts_DIRECTORY)
    if not os.path.exists(REWRITE_SUMMARIES_DIRECTORY):
        os.makedirs(REWRITE_SUMMARIES_DIRECTORY)

    base_name, _ = os.path.splitext(os.path.basename(INPUT_FILE_PATH))
    
    summary_file_name = f"{base_name}_rewrite_summary.txt"
    summary_file_path = os.path.join(REWRITE_SUMMARIES_DIRECTORY, summary_file_name)

    # This output_file_path is now more of a placeholder pattern for the main function to use
    # The actual name will be determined inside rewrite_srt_script_main based on the mode.
    output_file_name_placeholder = f"{base_name}_{output_filename_suffix_placeholder}_{timestamp_str}.srt" 
    output_file_path_placeholder = os.path.join(rewrite_scripts_DIRECTORY, output_file_name_placeholder)

    rewrite_srt_script_main(
        input_path=INPUT_FILE_PATH,
        output_path_placeholder=output_file_path_placeholder,
        model=REWRITE_MODEL_NAME,
        rewrite_instruction_block=SELECTED_BLOCK_REWRITE_INSTRUCTION, 
        batch_size=SRT_BATCH_SIZE,
        summary_model=SUMMARIZATION_MODEL_NAME,
        title_model=TITLE_DESCRIPTION_MODEL_NAME, # Pass title model
        description_model=TITLE_DESCRIPTION_MODEL_NAME, # Pass description model
        summary_file_path=summary_file_path,
        rewrite_mode=REWRITE_MODE,
        block_rewrite_style=BLOCK_REWRITE_STYLE
    ) 