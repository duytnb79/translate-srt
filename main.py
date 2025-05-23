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
    "gpt-4.1-mini": {"prompt": 0.00015, "completion": 0.00060}, 
    "gpt-3.5-turbo": {"prompt": 0.0000005, "completion": 0.0000015},
    "gpt-4o": {"prompt": 0.000005, "completion": 0.000015},
    "gpt-4o-mini": {"prompt": 0.00000015, "completion": 0.0000006},
    # Add other models and their pricing here as needed
    # Ensure these are $ per 1K tokens.
    # Example: gpt-4o-mini is $0.15 per 1M prompt tokens -> $0.15 / 1000 (1K tokens) = $0.00015
    # Example: gpt-4o-mini is $0.60 per 1M completion tokens -> $0.60 / 1000 (1K tokens) = $0.00060
    # So, for gpt-4o-mini, it should be:
    # "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.00060}, # Corrected based on $0.15/1M and $0.60/1M
} 
SUMMARIES_DIRECTORY = "summaries" # Directory to store summaries
SUMMARY_SECTION_MARKERS = {
    "summary": "[CONTEXTUAL SUMMARY]",
    "titles": "[SUGGESTED TITLES]",
    "description": "[SUGGESTED DESCRIPTION]"
}

# --- Helper Functions ---

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
    """Generates a contextual summary of the entire SRT content to aid Vietnamese translation."""
    print(f"\nĐang tạo tóm tắt ngữ cảnh với model: {model}...")
    full_text_for_summary = []
    for block_text in srt_blocks:
        lines = block_text.split("\n")
        if len(lines) >= 3:
            full_text_for_summary.append(" ".join(lines[2:]))
    
    if not full_text_for_summary:
        print("Không có nội dung để tạo tóm tắt.")
        return "", 0, 0

    combined_text = "\n---\n".join(full_text_for_summary)

    summary_prompt = (
        "Bạn là một trợ lý phân tích nội dung. Dựa vào toàn bộ nội dung kịch bản gốc được cung cấp dưới đây, "
        "hãy viết một bản tóm tắt bằng tiếng Việt (khoảng 200-300 từ). "
        "Bản tóm tắt này CẦN LÀM RÕ bối cảnh, cốt truyện chính, các nhân vật quan trọng, "
        "mối quan hệ và động cơ của họ, cũng như giọng điệu, không khí chung của kịch bản. "
        "ĐẶC BIỆT: Hãy liệt kê các nhân vật chính kèm theo danh xưng (ví dụ: Ông, Bà, Anh, Chị, Cậu, Cô, Bác sĩ, Giáo sư, Tướng quân) phù hợp với vai trò và mối quan hệ của họ trong kịch bản. "
        "Thông tin này sẽ được sử dụng để đảm bảo tính nhất quán trong bản dịch. "
        "Mục đích là để cung cấp thông tin nền cần thiết cho một dịch giả chuyên nghiệp dịch phụ đề sang tiếng Việt một cách chính xác, tự nhiên và bám sát tinh thần của tác phẩm. "
        "Tập trung vào việc tóm tắt các yếu tố trên.\n\nNỘI DUNG KỊCH BẢN GỐC:\n"
        f"{combined_text}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý phân tích nội dung chuyên nghiệp, có khả năng nắm bắt các yếu tố quan trọng của kịch bản để hỗ trợ dịch thuật. Chỉ cung cấp tóm tắt bằng tiếng Việt."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.5,
            max_tokens=600  # Increased slightly for potentially richer summary
        )
        summary = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        print("Đã tạo xong tóm tắt ngữ cảnh.")
        return summary.strip(), prompt_tokens, completion_tokens
    except Exception as e:
        print(f"Lỗi khi tạo tóm tắt ngữ cảnh: {e}")
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
            # Simple title parsing, assuming titles don't have numbering like "1. " in the file
            # If they do, it will be part of the title. Or strip leading digits/punctuation.
            if line.strip(): # Avoid empty lines
                 # Strip potential numbering like "1. ", "- ", etc.
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

def get_or_create_summary(srt_blocks: list[str], summary_model: str, title_model: str, description_model: str, summary_file_path: str) -> tuple[str, list[str], str, int, int]:
    """Loads summary, titles, and description from file if exists, otherwise generates and saves them."""
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if os.path.exists(summary_file_path):
        try:
            with open(summary_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            summary_text, titles_list, description_text = parse_structured_summary_content(file_content)
            
            if summary_text and titles_list and description_text: # Check if all parts were found
                print(f"Đã tải tóm tắt, tiêu đề và mô tả từ file: {summary_file_path}")
                return summary_text, titles_list, description_text, 0, 0 # No token cost for loading
            else:
                print(f"File tóm tắt {summary_file_path} không đầy đủ hoặc lỗi cấu trúc. Sẽ tạo mới.")
        except Exception as e:
            print(f"Lỗi khi đọc/phân tích file tóm tắt {summary_file_path}: {e}. Sẽ tạo mới.")
    
    # Generate summary
    summary_text, s_p, s_c = generate_contextual_summary(srt_blocks, summary_model)
    total_prompt_tokens += s_p
    total_completion_tokens += s_c

    titles_list = []
    description_text = ""

    if summary_text: # Only generate titles/description if summary was successful
        # Generate titles
        titles_list, t_p, t_c = generate_seo_video_title(summary_text, title_model)
        total_prompt_tokens += t_p
        total_completion_tokens += t_c
        
        # Generate description
        description_text, d_p, d_c = generate_seo_video_description(summary_text, description_model) # New call
        total_prompt_tokens += d_p
        total_completion_tokens += d_c
    else:
        print("Không thể tạo tóm tắt, bỏ qua việc tạo tiêu đề và mô tả.")

    if summary_text or titles_list or description_text: # Save if any part was generated
        try:
            os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
            with open(summary_file_path, "w", encoding="utf-8") as f:
                f.write(f"{SUMMARY_SECTION_MARKERS['summary']}\n{summary_text}\n\n")
                f.write(f"{SUMMARY_SECTION_MARKERS['titles']}\n")
                for i, title in enumerate(titles_list):
                    f.write(f"{i+1}. {title}\n") # Write with numbering for readability in file
                f.write("\n")
                f.write(f"{SUMMARY_SECTION_MARKERS['description']}\n{description_text}\n")
            print(f"Đã lưu tóm tắt, tiêu đề và mô tả mới vào file: {summary_file_path}")
        except Exception as e:
            print(f"Lỗi khi lưu file tóm tắt tổng hợp {summary_file_path}: {e}")
            
    return summary_text, titles_list, description_text, total_prompt_tokens, total_completion_tokens

def generate_seo_video_title(context_summary: str, model: str) -> tuple[list[str], int, int]:
    """Generates SEO-friendly and clickbait-y video titles based on the context summary."""
    print(f"Đang tạo tiêu đề SEO cho video với model: {model}...")
    
    if not context_summary:
        print("Không có tóm tắt ngữ cảnh để tạo tiêu đề. Bỏ qua bước này.")
        return [], 0, 0

    title_prompt = (
        "Bạn là một chuyên gia SEO và sáng tạo nội dung video. "
        "Dựa vào bản tóm tắt kịch bản được cung cấp dưới đây, hãy đề xuất 5 tiêu đề video bằng tiếng Việt. "
        "Các tiêu đề này cần phải:\\n"
        "1. Chuẩn SEO: Chứa từ khóa quan trọng có thể có trong tóm tắt.\\n"
        "2. Thu hút tò mò (clickbait): Kích thích người xem nhấp vào video.\\n"
        "3. Ngắn gọn và hấp dẫn: Dễ nhớ và gây ấn tượng.\\n"
        "4. Phù hợp với nội dung video được tóm tắt.\\n\\n"
        "TÓM TẮT KỊCH BẢN:\\n"
        f"\'\'\'{context_summary}\'\'\'\\n\\n"
        "Vui lòng chỉ cung cấp danh sách 5 tiêu đề, mỗi tiêu đề trên một dòng, không có đánh số hay ký tự đặc biệt ở đầu dòng. Ví dụ:\\n"
        "Tiêu đề mẫu 1\\n"
        "Tiêu đề mẫu 2\\n"
        "Tiêu đề mẫu 3\\n"
        "Tiêu đề mẫu 4\\n"
        "Tiêu đề mẫu 5"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia SEO và viết tiêu đề video clickbait bằng tiếng Việt. Chỉ trả về danh sách tiêu đề, mỗi tiêu đề một dòng."},
                {"role": "user", "content": title_prompt}
            ],
            temperature=0.7, 
            max_tokens=300 
        )
        
        titles_text = response.choices[0].message.content
        titles = [title.strip() for title in titles_text.split("\n") if title.strip()]
        
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        
        print("Đã tạo xong các đề xuất tiêu đề video.")
        return titles, prompt_tokens, completion_tokens
    except Exception as e:
        print(f"Lỗi khi tạo tiêu đề video: {e}")
        return [], 0, 0

def generate_seo_video_description(context_summary: str, model: str) -> tuple[str, int, int]:
    """Generates an SEO-friendly and engaging video description based on the context summary."""
    print(f"Đang tạo mô tả SEO cho video với model: {model}...")
    
    if not context_summary:
        print("Không có tóm tắt ngữ cảnh để tạo mô tả. Bỏ qua bước này.")
        return "", 0, 0

    description_prompt = (
        "Bạn là một chuyên gia SEO và copywriter cho nội dung video. "
        "Dựa vào bản tóm tắt kịch bản được cung cấp, hãy viết một mô tả video ngắn gọn (khoảng 100-150 từ) bằng tiếng Việt. "
        "Mô tả này cần phải:\n"
        "1. Chuẩn SEO: Chứa các từ khóa quan trọng có thể tìm thấy trong tóm tắt.\n"
        "2. Hấp dẫn: Tóm lược nội dung chính một cách lôi cuốn, kích thích người xem tìm hiểu thêm.\n"
        "3. Kêu gọi hành động (tùy chọn nhẹ nhàng): Có thể gợi ý người xem thích, chia sẻ hoặc bình luận nếu phù hợp.\n"
        "4. Phù hợp với nội dung video được tóm tắt.\n\n"
        "TÓM TẮT KỊCH BẢN:\n"
        f"'''{context_summary}'''\n\n"
        "Vui lòng chỉ cung cấp phần mô tả đã viết, không có tiêu đề hay giới thiệu nào khác cho phần mô tả này."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia SEO và viết mô tả video hấp dẫn bằng tiếng Việt. Chỉ trả về nội dung mô tả.",},
                {"role": "user", "content": description_prompt}
            ],
            temperature=0.6, 
            max_tokens=300 
        )
        
        description = response.choices[0].message.content.strip()
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        
        print("Đã tạo xong mô tả video.")
        return description, prompt_tokens, completion_tokens
    except Exception as e:
        print(f"Lỗi khi tạo mô tả video: {e}")
        return "", 0, 0

def translate_single_batch(batch_content: list[str], model: str, context_summary: str) -> tuple[list[str], int, int]:
    """
    Translates a single batch of SRT blocks using OpenAI API, with context summary.
    Aims for natural, contextually appropriate, and script-faithful Vietnamese translations.
    Returns the translated lines, prompt tokens, and completion tokens.
    """
    original_blocks_in_batch = []
    original_texts_for_prompt = []

    for block_index, block_text in enumerate(batch_content):
        lines = block_text.split("\n")
        if len(lines) < 3:
            continue
        index, timecode = lines[0], lines[1]
        text_to_translate = "\n".join(lines[2:])
        original_blocks_in_batch.append((index, timecode, text_to_translate))
        original_texts_for_prompt.append({"id": f"item_{block_index}", "text": text_to_translate})

    if not original_texts_for_prompt:
        return [], 0, 0

    prompt_text_parts = [f'{item["id"]}: "{item["text"]}"' for item in original_texts_for_prompt]
    prompt_text_for_api = "\n".join(prompt_text_parts) # Renamed for clarity

    system_message_content = (
        "Bạn là một dịch giả phụ đề tiếng Việt chuyên nghiệp và giàu kinh nghiệm. "
        "Nhiệm vụ của bạn là dịch các đoạn hội thoại sau sang tiếng Việt một cách TỰ NHIÊN, CHÍNH XÁC, PHÙ HỢP VỚI NGỮ CẢNH VÀ VĂN PHONG CỦA KỊCH BẢN. "
        "Hãy đảm bảo bản dịch truyền tải đúng ý nghĩa, sắc thái của lời thoại gốc và phù hợp để hiển thị dưới dạng phụ đề. "
        "QUAN TRỌNG: Nếu phát hiện tên riêng của nhân vật (ví dụ: John, Mary, Dr. Smith), hãy phiên âm chúng sang tiếng Việt một cách tự nhiên và thông dụng (ví dụ: Giôn, Mê-ri, Bác sĩ Smít). "
        "Hãy nhất quán với cách phiên âm tên trong toàn bộ bản dịch. "
        "ĐẶC BIỆT QUAN TRỌNG: Hãy tham khảo kỹ danh sách nhân vật và danh xưng được cung cấp trong phần TÓM TẮT KỊCH BẢN dưới đây và sử dụng chúng một cách nhất quán khi dịch. "
        "Sử dụng tóm tắt kịch bản dưới đây để hiểu rõ hơn về bối cảnh tổng thể:\n\n"
        f"TÓM TẮT KỊCH BẢN:\n'''{context_summary}'''\n\n"
        "Chỉ trả lời bằng một đối tượng JSON hợp lệ theo yêu cầu." 
    )

    user_prompt = (
        "Dựa vào vai trò và tóm tắt kịch bản đã được cung cấp, hãy dịch các đoạn hội thoại được đánh số sau sang tiếng Việt. "
        "LƯU Ý: Cố gắng bám sát nội dung và tinh thần của kịch bản gốc, đảm bảo lời thoại dịch ra tự nhiên và phù hợp với nhân vật. "
        "Đặc biệt chú ý phiên âm tên riêng của nhân vật sang tiếng Việt một cách nhất quán. "
        "Giữ nguyên số thứ tự và định dạng JSON như ví dụ. Tuyệt đối không thêm bất kỳ giải thích hay bình luận nào ngoài đối tượng JSON.\n\n"
        f"Đoạn hội thoại cần dịch:\n{prompt_text_for_api}\n\n"
        "Ví dụ định dạng JSON đầu ra: {\"item_0\": \"bản dịch cho item_0\", \"item_1\": \"bản dịch cho item_1\"}"
    )

    translated_srt_lines_for_batch = []
    prompt_tokens = 0
    completion_tokens = 0

    try:
        response = client.chat.completions.create(
                model=model,
                messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_prompt}
                ],
            temperature=0.2, # Slightly lower temperature for more deterministic and faithful translation
            max_tokens=3500, 
            response_format={"type": "json_object"}
        )

        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
        
        translated_content_json = response.choices[0].message.content
        if not translated_content_json:
             print("Lỗi: API trả về nội dung trống.")
             raise ValueError("Empty content from API")

        translated_content = json.loads(translated_content_json)

        for i, (idx, timecode, _) in enumerate(original_blocks_in_batch):
            prompt_item_id = f"item_{i}"
            translated_line = translated_content.get(prompt_item_id, "[LỖI DỊCH - KHÔNG TÌM THẤY ID HOẶC LỖI PHÂN TÍCH]") # Modified error message
            translated_srt_lines_for_batch.append(f"{idx}\n{timecode}\n{translated_line}")

    except json.JSONDecodeError as e:
        print(f"Lỗi khi giải mã JSON: {e}. Dữ liệu nhận được: {translated_content_json if 'translated_content_json' in locals() else 'Không có dữ liệu'}")
        for idx, timecode, _ in original_blocks_in_batch:
            translated_srt_lines_for_batch.append(f"{idx}\n{timecode}\n[LỖI GIẢI MÃ JSON TỪ API]") # Modified error message
    except Exception as e:
        print(f"Lỗi khi dịch batch: {e}")
        for idx, timecode, _ in original_blocks_in_batch:
            translated_srt_lines_for_batch.append(f"{idx}\n{timecode}\n[LỖI DỊCH API KHÔNG XÁC ĐỊNH]") # Modified error message
            
    return translated_srt_lines_for_batch, prompt_tokens, completion_tokens

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
    """Saves the translated results to the specified output file."""
    # print(f"Đang ghi kết quả vào file: {output_path}") # Can be un-commented for verbose logging
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(results))

# --- Main Translation Function ---

def batch_translate_srt_main(input_path: str, output_path: str, model: str, batch_size: int, summary_model: str, title_generation_model: str, description_generation_model: str, summary_file_path: str):
    """Main function to translate SRT file, generate summary, title, description and calculate cost."""
    srt_blocks = load_srt_blocks(input_path)
    if not srt_blocks:
        print("Không có nội dung để xử lý. Kết thúc.")
        return

    grand_total_prompt_tokens = 0
    grand_total_completion_tokens = 0
    
    print("\n--- Bước chuẩn bị: Tạo/Tải tóm tắt, tiêu đề & mô tả ---")
    context_summary, video_titles, video_description, s_t_d_p_tokens, s_t_d_c_tokens = get_or_create_summary(
        srt_blocks, 
        summary_model, 
        title_generation_model, 
        description_generation_model, # Pass new model
        summary_file_path
    )
    grand_total_prompt_tokens += s_t_d_p_tokens
    grand_total_completion_tokens += s_t_d_c_tokens

    if not context_summary:
        print("CẢNH BÁO: Không thể tạo hoặc tải tóm tắt. Các bước tiếp theo có thể bị ảnh hưởng hoặc bỏ qua.")
    else:
        print("\n--- Thông tin bổ trợ đã tạo/tải ---")
        print(f"Tóm tắt ngữ cảnh:\n{context_summary}")
        if video_titles:
            print("\nCác đề xuất tiêu đề video:")
            for i, title in enumerate(video_titles):
                print(f"  {i+1}. {title}")
        else:
            print("\nKhông có tiêu đề video nào được tạo/tải.")
        
        if video_description:
            print(f"\nMô tả video đề xuất:\n{video_description}")
        else:
            print("\nKhông có mô tả video nào được tạo/tải.")

    # --- Step 2 (User-facing): Batch Translation with Context ---
    print(f"\n--- Bước 2: Dịch thuật file {input_path} ---")
    print(f"Sử dụng model: {model}, batch size: {batch_size}")
    if context_summary:
        # print("Sử dụng tóm tắt ngữ cảnh để dịch.") # Already clear from previous steps
        pass
    else:
        print("CẢNH BÁO: Dịch thuật mà không có tóm tắt ngữ cảnh bổ sung (nếu có).")

    batches = [srt_blocks[i:i + batch_size] for i in range(0, len(srt_blocks), batch_size)]
    all_translated_results = []
    
    # Keep track of tokens for translation *only* within this section
    translation_section_p_tokens = 0
    translation_section_c_tokens = 0

    for batch_content in tqdm(batches, desc="Đang dịch các nhóm", unit="nhóm"):
        translated_lines_for_batch, batch_p_tokens, batch_c_tokens = translate_single_batch(batch_content, model, context_summary if context_summary else "") # Pass empty string if no summary
        
        all_translated_results.extend(translated_lines_for_batch)
        translation_section_p_tokens += batch_p_tokens
        translation_section_c_tokens += batch_c_tokens
        
        save_results_to_file(output_path, all_translated_results)
        time.sleep(1.5)
    
    grand_total_prompt_tokens += translation_section_p_tokens
    grand_total_completion_tokens += translation_section_c_tokens

    # Cost Calculation
    if s_t_d_p_tokens > 0 or s_t_d_c_tokens > 0: 
        # Cost for summary, titles, and description generation (if API was called)
        # We use summary_model for cost calculation here as an approximation, 
        # or we could break down costs by each model if pricing differs significantly and precisely.
        # For simplicity, if different models used for summary/title/desc have similar pricing, one is okay.
        # Otherwise, get_or_create_summary should return detailed token counts per model.
        # For now, assuming summary_model is representative for the combined cost of s_t_d tokens.
        combined_summary_cost = calculate_cost(s_t_d_p_tokens, s_t_d_c_tokens, summary_model) 
        print(f"\nChi phí tóm tắt, tiêu đề & mô tả (ước tính với model chính: {summary_model}): ${combined_summary_cost:.6f}")
        # Removed individual title_generation_cost print as it's now part of combined cost

    translation_cost = calculate_cost(translation_section_p_tokens, translation_section_c_tokens, model)
    print(f"Chi phí dịch thuật (model: {model}): ${translation_cost:.6f}")
    
    total_estimated_cost = combined_summary_cost + translation_cost # Updated total cost calculation

    print("\n--- Hoàn Thành Xử Lý ---")
    print(f"Kết quả dịch đã được lưu tại: {output_path}")
    # Video titles and description are printed earlier if available
    print(f"Tổng số prompt tokens (bao gồm tóm tắt, tiêu đề, mô tả, dịch thuật): {grand_total_prompt_tokens}")
    print(f"Tổng số completion tokens (bao gồm tóm tắt, tiêu đề, dịch thuật): {grand_total_completion_tokens}")
    print(f"Tổng ước tính chi phí: ${total_estimated_cost:.6f}")


# --- Script Execution ---

if __name__ == "__main__":
    INPUT_FILE_PATH = "data/0522.srt"
    OUTPUT_DIRECTORY = "output"
    # SUMMARIES_DIRECTORY is defined globally
    TRANSLATION_MODEL_NAME = "gpt-4o-mini"
    SUMMARIZATION_MODEL_NAME = "gpt-4o-mini" 
    TITLE_GENERATION_MODEL_NAME = "gpt-4o-mini" 
    BATCH_SIZE = 20
    # Generate a timestamp string for filenames
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    # Ensure output and summaries directories exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    if not os.path.exists(SUMMARIES_DIRECTORY):
        os.makedirs(SUMMARIES_DIRECTORY)

    base_name, _ = os.path.splitext(os.path.basename(INPUT_FILE_PATH))
    
    # Define summary file path
    summary_file_name = f"{base_name}_summary.txt"
    summary_file_path = os.path.join(SUMMARIES_DIRECTORY, summary_file_name)

    # Define output file path using timestamp
    translated_file_ext = ".srt"
    output_file_name = f"{base_name}_translated_batch_{timestamp_str}{translated_file_ext}" 
    output_file_path = os.path.join(OUTPUT_DIRECTORY, output_file_name)

    batch_translate_srt_main(
        input_path=INPUT_FILE_PATH,
        output_path=output_file_path,
        model=TRANSLATION_MODEL_NAME,
        batch_size=BATCH_SIZE,
        summary_model=SUMMARIZATION_MODEL_NAME,
        title_generation_model=TITLE_GENERATION_MODEL_NAME,
        description_generation_model=TITLE_GENERATION_MODEL_NAME, # Using same model for description for now
        summary_file_path=summary_file_path
    )
