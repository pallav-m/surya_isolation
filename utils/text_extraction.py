# surya_isolation.utils/text_extraction.py

def extract_text_from_rec_result(rec_result):
    """Add combined_text key to recognition result to get combined lines as a string"""
    for result in rec_result:
        result_text_lines = result['text_lines']
        # Clean up character info to save space:
        for text_line in result_text_lines:
            del text_line['chars']
        lines = []
        for text_line in result_text_lines:
            lines.append(text_line['text'])
        result.update({"combined_text": "\n".join(lines)})
