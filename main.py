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

def get_or_create_summary(srt_blocks: list[str], summary_model: str, summary_file_path: str) -> tuple[str, int, int]:
    """Loads summary from file if exists, otherwise generates and saves it."""
    if os.path.exists(summary_file_path):
        try:
            with open(summary_file_path, "r", encoding="utf-8") as f:
                summary_text = f.read()
            print(f"Đã tải tóm tắt ngữ cảnh từ file: {summary_file_path}")
            return summary_text, 0, 0 # No token cost for loading from file
        except Exception as e:
            print(f"Lỗi khi đọc file tóm tắt {summary_file_path}: {e}. Sẽ tạo tóm tắt mới.")
    
    # If file doesn't exist or reading failed, generate new summary
    summary_text, p_tokens, c_tokens = generate_contextual_summary(srt_blocks, summary_model)
    
    if summary_text: # Save the newly generated summary if it's not empty
        try:
            os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
            with open(summary_file_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            print(f"Đã lưu tóm tắt mới vào file: {summary_file_path}")
        except Exception as e:
            print(f"Lỗi khi lưu file tóm tắt {summary_file_path}: {e}")
            
    return summary_text, p_tokens, c_tokens

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

def batch_translate_srt_main(input_path: str, output_path: str, model: str, batch_size: int, summary_model: str, summary_file_path: str):
    """Main function to translate SRT file in batches and calculate cost, including summarization step."""
    srt_blocks = load_srt_blocks(input_path)
    if not srt_blocks:
        print("Không có nội dung để dịch. Kết thúc.")
        return

    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Step 1: Get or Create Contextual Summary
    context_summary, summary_p_tokens, summary_c_tokens = get_or_create_summary(srt_blocks, summary_model, summary_file_path)
    total_prompt_tokens += summary_p_tokens
    total_completion_tokens += summary_c_tokens

    # Step 2: Batch Translation with Context
    batches = [srt_blocks[i:i + batch_size] for i in range(0, len(srt_blocks), batch_size)]
    all_translated_results = []

    print(f"\nBắt đầu dịch file: {input_path} với model {model}, batch size: {batch_size}")
    if context_summary:
        # print("Sử dụng tóm tắt ngữ cảnh để dịch.") # Already printed in get_or_create_summary
        pass
    else:
        print("Không có tóm tắt ngữ cảnh hoặc có lỗi khi tạo, tiếp tục dịch không có ngữ cảnh bổ sung.")

    for batch_content in tqdm(batches, desc="Đang dịch các nhóm", unit="nhóm"):
        translated_lines_for_batch, p_tokens, c_tokens = translate_single_batch(batch_content, model, context_summary)
        
        all_translated_results.extend(translated_lines_for_batch)
        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens
        
        save_results_to_file(output_path, all_translated_results)
        time.sleep(1.5)

    # Cost Calculation
    summary_cost = 0.0
    if summary_p_tokens > 0 or summary_c_tokens > 0: # Cost for summary only if it was generated via API
        summary_cost = calculate_cost(summary_p_tokens, summary_c_tokens, summary_model)
        print(f"Chi phí tóm tắt (model: {summary_model}): ${summary_cost:.6f}")

    # Calculate translation cost (excluding summary tokens if they were from API for this run)
    translation_p_tokens = total_prompt_tokens - summary_p_tokens
    translation_c_tokens = total_completion_tokens - summary_c_tokens
    translation_cost = calculate_cost(translation_p_tokens, translation_c_tokens, model)
    print(f"Chi phí dịch thuật (model: {model}): ${translation_cost:.6f}")
    
    total_estimated_cost = summary_cost + translation_cost

    print("\n--- Hoàn Thành Dịch ---")
    print(f"Kết quả đã được lưu tại: {output_path}")
    print(f"Tổng số prompt tokens (bao gồm tóm tắt nếu tạo mới): {total_prompt_tokens}")
    print(f"Tổng số completion tokens (bao gồm tóm tắt nếu tạo mới): {total_completion_tokens}")
    print(f"Tổng ước tính chi phí: ${total_estimated_cost:.6f}")


# --- Script Execution ---

if __name__ == "__main__":
    INPUT_FILE_PATH = "data/0522.srt"
    OUTPUT_DIRECTORY = "output"
    # SUMMARIES_DIRECTORY is defined globally
    TRANSLATION_MODEL_NAME = "gpt-4o-mini"
    SUMMARIZATION_MODEL_NAME = "gpt-4o-mini" 
    BATCH_SIZE = 20
    RUN_UUID = str(uuid.uuid4())

    # Ensure output and summaries directories exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    if not os.path.exists(SUMMARIES_DIRECTORY):
        os.makedirs(SUMMARIES_DIRECTORY)

    base_name, _ = os.path.splitext(os.path.basename(INPUT_FILE_PATH))
    
    # Define summary file path
    summary_file_name = f"{base_name}_summary.txt"
    summary_file_path = os.path.join(SUMMARIES_DIRECTORY, summary_file_name)

    # Define output file path using UUID
    translated_file_ext = ".srt"
    output_file_name = f"{base_name}_translated_batch_{RUN_UUID}{translated_file_ext}" 
    output_file_path = os.path.join(OUTPUT_DIRECTORY, output_file_name)

    batch_translate_srt_main(
        input_path=INPUT_FILE_PATH,
        output_path=output_file_path,
        model=TRANSLATION_MODEL_NAME,
        batch_size=BATCH_SIZE,
        summary_model=SUMMARIZATION_MODEL_NAME,
        summary_file_path=summary_file_path
    )
