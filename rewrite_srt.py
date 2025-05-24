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

# --- Constants for Prompt Files ---
PROMPTS_DIR = "prompts"
PERSONAS_SUB_DIR = "personas" # New constant for personas subdirectory

# --- Helper Function to Load Prompts ---
def load_prompt(file_name: str, persona_name: str | None = None) -> str:
    """Loads a prompt from the specified file in the PROMPTS_DIR or a persona-specific subdirectory."""
    if persona_name:
        file_path = os.path.join(PROMPTS_DIR, PERSONAS_SUB_DIR, persona_name, file_name)
    else:
        file_path = os.path.join(PROMPTS_DIR, file_name)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file prompt: {file_path}. Trả về chuỗi trống.")
        return ""
    except Exception as e:
        print(f"Lỗi khi đọc file prompt {file_path}: {e}. Trả về chuỗi trống.")
        return ""

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

    base_summary_prompt_template = load_prompt("contextual_summary_prompt.txt")
    if not base_summary_prompt_template:
        print("LỖI: Không thể tải mẫu prompt tóm tắt. Bỏ qua tạo tóm tắt.")
        return "", 0, 0
    
    summary_prompt = base_summary_prompt_template.format(combined_text=combined_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý phân tích nội dung chuyên nghiệp, có khả năng nắm bắt các yếu tố quan trọng của kịch bản để hỗ trợ việc viết lại. Chỉ cung cấp tóm tắt bằng tiếng Việt."},
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
    
    base_titles_prompt_template = load_prompt("creative_titles_prompt.txt")
    if not base_titles_prompt_template:
        print("LỖI: Không thể tải mẫu prompt tiêu đề. Bỏ qua tạo tiêu đề.")
        return [], 0, 0

    prompt = base_titles_prompt_template.format(context_summary=context_summary)

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

    base_description_prompt_template = load_prompt("engaging_description_prompt.txt")
    if not base_description_prompt_template:
        print("LỖI: Không thể tải mẫu prompt mô tả. Bỏ qua tạo mô tả.")
        return "", 0, 0
        
    prompt = base_description_prompt_template.format(context_summary=context_summary)

    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": "Chuyên gia viết caption TikTok tiếng Việt siêu thu hút, ngắn gọn, kèm hashtag liên quan."}, {"role": "user", "content": prompt}],
            temperature=0.75, # Maintain creativity for description
            max_tokens=200 # Descriptions are short
        )
        return response.choices[0].message.content.strip(), response.usage.prompt_tokens if response.usage else 0, response.usage.completion_tokens if response.usage else 0
    except Exception as e:
        print(f"Lỗi tạo mô tả (viết lại, TikTok): {e}"); return "", 0, 0

def generate_new_narrative_segments(context_summary: str, original_srt_full_text: str, target_duration_seconds: float, model: str, narrative_goal_prompt_template: str) -> tuple[list[str] | list[dict], int, int]:
    """
    Generates new narrative segments based on summary, full original text, target duration, 
    and a goal prompt template.
    For themed narrative, returns a list of dicts: [{'title': str, 'segments': list[str]}].
    Otherwise, returns a list of strings.
    """
    print(f"\nĐang tạo các phân đoạn tự sự mới với model: {model}...")
    
    if not context_summary and not original_srt_full_text:
        print("Lỗi: Cần tóm tắt ngữ cảnh hoặc toàn bộ kịch bản gốc để tạo tự sự mới.")
        return [], 0, 0

    # The prompt template is now expected to be fully formatted by the caller for themed narrative.
    # For old style, it might still use duration_guidance. This function becomes more of a wrapper.
    # The narrative_goal_prompt_template passed in IS the full_prompt's main body.

    # The main template is loaded, and placeholders for summary/original_text are appended later.
    # The {story_completeness_guidance} is part of narrative_goal_prompt_template itself loaded from file.
    narrative_goal_prompt_body = narrative_goal_prompt_template # This is already the formatted string passed in.

    full_prompt = (
        f"{narrative_goal_prompt_body}\n\n" 
        "TÓM TẮT KỊCH BẢN ĐỂ THAM KHẢO (cho bối cảnh và giọng điệu tổng thể):\n"
        f"'''{context_summary}'''\\n\\n"
        "KỊCH BẢN GỐC ĐẦY ĐỦ (để đảm bảo bám sát các sự kiện và thông tin chi tiết):\n"
        f"'''{original_srt_full_text}'''"
        "Toàn bộ câu chuyện và tiêu đề các chủ đề cần được tối ưu để thu hút người nghe và thân thiện với SEO."
    )

    prompt_tokens = 0
    completion_tokens = 0
    # segments = [] # Old return type
    parsed_output = [] # Can be list of strings or list of dicts

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là một nhà văn sáng tạo, chuyên gia xây dựng kịch bản tự sự hấp dẫn bằng tiếng Việt. Nhiệm vụ của bạn là tạo ra các đoạn văn bản tường thuật dựa trên tóm tắt và hướng dẫn được cung cấp. Hãy tuân thủ định dạng đầu ra được yêu cầu trong hướng dẫn."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.75, 
            max_tokens=3800  # Increased slightly for potentially longer themed output
        )

        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
        
        raw_response_content = response.choices[0].message.content
        if raw_response_content:
            # Try parsing for new themed format first
            if "||NEW_THEME_TITLE||" in raw_response_content and "||END_THEME||" in raw_response_content:
                themed_data = []
                theme_blocks = raw_response_content.split("||END_THEME||")
                for block in theme_blocks:
                    block = block.strip()
                    if not block.startswith("||NEW_THEME_TITLE||"):
                        continue
                    
                    parts = block.split("\n", 1) # CORRECTED: Split title line from content by newline
                    title_line = parts[0].replace("||NEW_THEME_TITLE||", "").strip()
                    
                    theme_segments_str = parts[1] if len(parts) > 1 else ""
                    segments_in_theme = [s.strip() for s in theme_segments_str.split("||NEW_SEGMENT||") if s.strip()]
                    
                    if title_line and segments_in_theme:
                        themed_data.append({"title": title_line, "segments": segments_in_theme})
                
                if themed_data:
                    parsed_output = themed_data
                    print(f"Đã tạo và phân tích được {len(themed_data)} chủ đề tự sự.")
                else:
                    print(f"Cảnh báo: Phát hiện các dấu hiệu định dạng chủ đề nhưng không phân tích được chủ đề nào. Nội dung thô: {raw_response_content[:300]}...")
                    # Fallback to old parsing if themed parsing fails but markers were present
                    parsed_output = [seg.strip() for seg in raw_response_content.split("||NEW_SEGMENT||") if seg.strip()]
                    if parsed_output:
                         print(f"Cảnh báo: Phân tích theo chủ đề thất bại, quay lại phân tích theo phân đoạn đơn lẻ: tìm thấy {len(parsed_output)} phân đoạn.")
                    else:
                        print(f"Lỗi: API trả về nội dung nhưng không tìm thấy phân đoạn hợp lệ sau khi tách bằng '||NEW_SEGMENT||' (thất bại cả themed parse). Nội dung thô: {raw_response_content[:300]}...")

            else: # Fallback to old segment parsing if no theme markers
                parsed_output = [seg.strip() for seg in raw_response_content.split("||NEW_SEGMENT||") if seg.strip()]
                if not parsed_output:
                 print(f"Cảnh báo: API trả về nội dung nhưng không tìm thấy phân đoạn hợp lệ sau khi tách bằng '||NEW_SEGMENT||'. Nội dung thô: {raw_response_content[:200]}...")
        else:
            print("Lỗi: API trả về nội dung trống cho việc tạo tự sự mới.")

    except Exception as e:
        print(f"Lỗi khi tạo phân đoạn tự sự mới: {e}")
        # parsed_output will remain empty

    if parsed_output:
        if isinstance(parsed_output, list) and parsed_output and isinstance(parsed_output[0], dict):
             print(f"Đã tạo được {len(parsed_output)} chủ đề tự sự mới.")
        elif isinstance(parsed_output, list) and parsed_output : # list of strings
            print(f"Đã tạo được {len(parsed_output)} phân đoạn tự sự mới (kiểu cũ).")
        # else it's empty, handled by the 'else' below
    else:
        print("Không tạo được phân đoạn/chủ đề tự sự nào.")
        
    return parsed_output, prompt_tokens, completion_tokens

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
        f"{rewrite_instruction}\\n\\n"
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
        f"{rewrite_instruction}\\n\\n"
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
    block_rewrite_style: str, # "formal" or "creative_text"
    narrative_persona: str | None = None # New parameter for persona
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

        actual_output_filename = f"{original_base_name}_themed_narrative_{run_identifier}.txt" # Changed to .txt
        actual_output_path = os.path.join(output_directory, actual_output_filename)
        
        print(f"Sử dụng model tạo tự sự: {model}")
        if narrative_persona:
            print(f"Sử dụng persona kể chuyện: {narrative_persona}")

        story_completeness_guidance = (
            "Câu chuyện kể lại cần bao quát đầy đủ nội dung chính của kịch bản gốc. "
            "Hãy chia câu chuyện thành ít nhất 3 chủ đề (và có thể nhiều hơn nếu nội dung gốc phong phú và cho phép) để truyền tải toàn bộ diễn biến một cách mạch lạc và hấp dẫn, với mỗi chủ đề có thể được phát triển thành một video riêng biệt. "
            "Toàn bộ câu chuyện và tiêu đề các chủ đề cần được tối ưu để thu hút người nghe và thân thiện với SEO."
        )

        # Load the base narrative prompt from the persona-specific file
        base_narrative_prompt_template = ""
        if narrative_persona:
            base_narrative_prompt_template = load_prompt("narrative_regeneration_prompt.txt", persona_name=narrative_persona)
        
        if not base_narrative_prompt_template:
            print(f"LỖI: Không thể tải mẫu prompt tạo tự sự cho persona '{narrative_persona if narrative_persona else 'default'}'. Bỏ qua tạo tự sự.")
            # Log cost even if prompt loading fails, as summary might have incurred cost
            print(f"Tổng số prompt tokens (bao gồm tóm tắt): {grand_total_prompt_tokens}")
            print(f"Tổng số completion tokens (bao gồm tóm tắt): {grand_total_completion_tokens}")
            print(f"Tổng ước tính chi phí cho phiên làm việc (chỉ tóm tắt): ${total_estimated_cost:.6f}")
            return

        narrative_goal_prompt_template = base_narrative_prompt_template # This now comes from the file

        original_srt_full_text = "\n\n".join(srt_blocks_full) 
        
        print("Đang tạo tự sự mới theo chủ đề...")
        
        themed_narratives, nr_p_tokens, nr_c_tokens = generate_new_narrative_segments(
            context_summary if context_summary else "", 
            original_srt_full_text,
            0, 
            model, 
            narrative_goal_prompt_template.replace("{story_completeness_guidance}", story_completeness_guidance)
        )

        grand_total_prompt_tokens += nr_p_tokens
        grand_total_completion_tokens += nr_c_tokens
        narrative_api_call_cost = calculate_cost(nr_p_tokens, nr_c_tokens, model)
        print(f"Chi phí API cho việc cố gắng tạo tự sự (model: {model}): ${narrative_api_call_cost:.6f}")
        total_estimated_cost += narrative_api_call_cost

        if themed_narratives and isinstance(themed_narratives, list) and len(themed_narratives) > 0 and isinstance(themed_narratives[0], dict):
            full_text_output = ""
            for theme_data in tqdm(themed_narratives, desc="Đang định dạng các chủ đề", unit="chủ đề"):
                full_text_output += f"{theme_data.get('title', 'KHÔNG CÓ TIÊU ĐỀ')}\n"
                for segment_text in theme_data.get('segments', []):
                    full_text_output += f"{segment_text}\n"
                full_text_output += "\n" # Blank line after each theme's segments
            
            if full_text_output.strip(): # Ensure there's actual content to save
                try:
                    os.makedirs(os.path.dirname(actual_output_path), exist_ok=True)
                    with open(actual_output_path, "w", encoding="utf-8") as f:
                        f.write(full_text_output.strip())
                    print(f"Kịch bản tự sự theo chủ đề đã được lưu tại: {actual_output_path}")
                except Exception as e:
                    print(f"Lỗi khi lưu file văn bản tự sự {actual_output_path}: {e}")
            else:
                print("Nội dung tự sự theo chủ đề được tạo ra trống sau khi định dạng. Sẽ không lưu file.")
        else:
            print("Không tạo được nội dung tự sự theo chủ đề như mong muốn (hoặc định dạng không đúng). File sẽ không được lưu.")
            # Cost of API call is already accounted for. No file is saved.

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
    REWRITE_MODE = "narrative_regeneration"  # Options: "narrative_regeneration", "block_by_block_rewrite"
    # -------------------------------

    # --- CHOOSE NARRATIVE PERSONA (Only for REWRITE_MODE = "narrative_regeneration") ---
    NARRATIVE_PERSONA = "BaDieuAn" # Set to None or other persona name as needed
    # ------------------------------------------------------------------------------------

    # --- CHOOSE REWRITE STYLE (Only for REWRITE_MODE = "block_by_block_rewrite") ---
    BLOCK_REWRITE_STYLE = "creative_text" # Options: "formal", "creative_text"
    # ----------------------------------------------------------------------------

    # Load block rewrite prompts from files
    REWRITE_INSTRUCTION_FORMAL = load_prompt("block_rewrite_formal_prompt.txt")
    REWRITE_INSTRUCTION_CREATIVE_TEXT = load_prompt("block_rewrite_creative_text_prompt.txt")

    SELECTED_BLOCK_REWRITE_INSTRUCTION = "" # Default to empty if loading fails
    output_filename_suffix_placeholder = "placeholder" # This will be refined by the main function

    if BLOCK_REWRITE_STYLE == "formal":
        if REWRITE_INSTRUCTION_FORMAL: # Check if prompt loaded successfully
            SELECTED_BLOCK_REWRITE_INSTRUCTION = REWRITE_INSTRUCTION_FORMAL
        else:
            print("LỖI: Không thể tải prompt viết lại formal. Sẽ sử dụng hướng dẫn trống.")
    elif BLOCK_REWRITE_STYLE == "creative_text":
        if REWRITE_INSTRUCTION_CREATIVE_TEXT: # Check if prompt loaded successfully
            SELECTED_BLOCK_REWRITE_INSTRUCTION = REWRITE_INSTRUCTION_CREATIVE_TEXT
        else:
            print("LỖI: Không thể tải prompt viết lại creative. Sẽ sử dụng hướng dẫn trống.")
    else:
        print(f"CẢNH BÁO: Kiểu viết lại khối '{BLOCK_REWRITE_STYLE}' không hợp lệ. Hướng dẫn viết lại có thể trống.")
    
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
        block_rewrite_style=BLOCK_REWRITE_STYLE,
        narrative_persona=NARRATIVE_PERSONA # Pass the persona
    ) 