#!/usr/bin/env python3
"""
Markdown Document Cleaner

This script takes JSON output from the strikethrough detector and removes
the identified strikethrough text from a Markdown document.

Requirements:
    None (uses only standard library)

Usage:
    Set the JSON_FILE_PATH and MARKDOWN_FILE_PATH variables and run the script.
"""

import json
import os
import re
from typing import List, Dict, Tuple

# Set your file paths here
JSON_FILE_PATH = "docling_converter/strikes_json/_Azərbaycan_Respublikasının_Torpaq_Məcəlləsi_strikethrough_report.json"  # Path to JSON output from strikethrough detector
MARKDOWN_FILE_PATH = "docling_converter/mecelleler-raw/mecelleler-final/AZƏRBAYCAN RESPUBLİKASININ TORPAQ MƏCƏLLƏSİ.md"  # Path to the Markdown document to clean
OUTPUT_FILE_PATH = "docling_converter/mecelleler-raw/mecelleler-final/cleaned_document.md"  # Path for the cleaned output


class MarkdownCleaner:
    def __init__(self):
        self.cleaning_log = []
    
    def load_json_report(self, json_file_path: str) -> List[Dict]:
        """
        Load the JSON report from the strikethrough detector.
        
        Args:
            json_file_path (str): Path to the JSON report file
            
        Returns:
            List[Dict]: List of strikethrough detection results
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Loaded JSON report from: {json_file_path}")
            return data
            
        except FileNotFoundError:
            print(f"Error: JSON file '{json_file_path}' not found.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in '{json_file_path}': {str(e)}")
            return []
        except Exception as e:
            print(f"Error loading JSON file: {str(e)}")
            return []
    
    def load_markdown_file(self, markdown_file_path: str) -> str:
        """
        Load the Markdown document to be cleaned.
        
        Args:
            markdown_file_path (str): Path to the Markdown file
            
        Returns:
            str: Content of the Markdown file
        """
        try:
            with open(markdown_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"Loaded Markdown file: {markdown_file_path}")
            print(f"File size: {len(content)} characters")
            return content
            
        except FileNotFoundError:
            print(f"Error: Markdown file '{markdown_file_path}' not found.")
            return ""
        except Exception as e:
            print(f"Error loading Markdown file: {str(e)}")
            return ""
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better matching by removing extra whitespace
        and normalizing line endings.
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Normalized text
        """
        # Normalize line endings and remove extra whitespace
        text = re.sub(r'\r\n|\r|\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def find_paragraph_in_markdown(self, markdown_content: str, paragraph_text: str) -> List[Tuple[int, int, str]]:
        """
        Find paragraph text in the Markdown document using exact matching only.
        
        Args:
            markdown_content (str): The Markdown document content
            paragraph_text (str): The paragraph text to find
            
        Returns:
            List[Tuple[int, int, str]]: List of (start_pos, end_pos, matched_text) tuples
        """
        matches = []
        
        if not paragraph_text:
            return matches
        
        # Try exact match first
        if paragraph_text in markdown_content:
            start_pos = markdown_content.find(paragraph_text)
            end_pos = start_pos + len(paragraph_text)
            matches.append((start_pos, end_pos, paragraph_text))
            return matches
        
        # Try normalized exact match
        normalized_search = self.normalize_text(paragraph_text)
        normalized_content = self.normalize_text(markdown_content)
        
        if normalized_search in normalized_content:
            # Find the position in normalized content
            norm_start = normalized_content.find(normalized_search)
            # Try to map back to original content (approximate)
            words_before = len(normalized_content[:norm_start].split())
            original_words = markdown_content.split()
            
            if words_before < len(original_words):
                # Reconstruct the approximate position
                estimated_start = len(' '.join(original_words[:words_before]))
                # Look for the text around this position
                search_start = max(0, estimated_start - 100)
                search_end = min(len(markdown_content), estimated_start + len(paragraph_text) + 100)
                search_section = markdown_content[search_start:search_end]
                
                if paragraph_text in search_section:
                    local_start = search_section.find(paragraph_text)
                    actual_start = search_start + local_start
                    actual_end = actual_start + len(paragraph_text)
                    matches.append((actual_start, actual_end, paragraph_text))
                    return matches
        
        # No exact match found
        return matches
    
    def _fuzzy_match(self, search_text: str, paragraph_text: str, threshold: float = 0.9) -> bool:
        """
        Perform fuzzy matching to handle slight text differences.
        
        Args:
            search_text (str): Text to search for
            paragraph_text (str): Text to search in
            threshold (float): Similarity threshold (0.0 to 1.0)
            
        Returns:
            bool: True if texts are similar enough
        """
        # Simple fuzzy matching based on common words
        search_words = set(search_text.lower().split())
        paragraph_words = set(paragraph_text.lower().split())
        
        if not search_words:
            return False
        
        common_words = search_words.intersection(paragraph_words)
        similarity = len(common_words) / len(search_words)
        
        return similarity >= threshold
    
    def remove_text_from_paragraph(self, paragraph: str, text_to_remove: str) -> str:
        """
        Remove specific text from a paragraph.
        
        Args:
            paragraph (str): The paragraph to clean
            text_to_remove (str): The text to remove
            
        Returns:
            str: The cleaned paragraph
        """
        if not text_to_remove.strip():
            return paragraph
        
        # Try exact match first
        if text_to_remove in paragraph:
            cleaned = paragraph.replace(text_to_remove, '')
        else:
            # Try normalized match
            normalized_remove = self.normalize_text(text_to_remove)
            normalized_paragraph = self.normalize_text(paragraph)
            
            if normalized_remove in normalized_paragraph:
                # Find the text in the original paragraph
                words_to_remove = normalized_remove.split()
                paragraph_words = paragraph.split()
                
                # Try to find and remove the sequence of words
                cleaned = self._remove_word_sequence(paragraph, words_to_remove)
            else:
                # If no match found, return original
                cleaned = paragraph
                self.cleaning_log.append({
                    'status': 'warning',
                    'message': f"Could not find exact text to remove: '{text_to_remove[:50]}...'"
                })
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _remove_word_sequence(self, text: str, words_to_remove: List[str]) -> str:
        """
        Remove a sequence of words from text.
        
        Args:
            text (str): The text to clean
            words_to_remove (List[str]): List of words to remove in sequence
            
        Returns:
            str: The cleaned text
        """
        # Create a pattern to match the word sequence with flexible whitespace
        pattern = r'\s*'.join(re.escape(word) for word in words_to_remove)
        cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return cleaned
    
    def _group_removals_by_paragraph(self, json_data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group all text removals by their source paragraph to handle multiple removals efficiently.
        
        Args:
            json_data (List[Dict]): Strikethrough detection results
            
        Returns:
            Dict[str, List[Dict]]: Dictionary mapping paragraph text to list of removal items
        """
        paragraph_removals = {}
        
        for file_result in json_data:
            if 'error' in file_result:
                continue
            
            strikethrough_items = file_result.get('strikethrough_items', [])
            
            for item in strikethrough_items:
                paragraph_text = item.get('paragraph_text', '') or item.get('cell_text', '')
                text_to_remove = item.get('text', '')
                
                if not paragraph_text or not text_to_remove:
                    continue
                
                if paragraph_text not in paragraph_removals:
                    paragraph_removals[paragraph_text] = []
                
                paragraph_removals[paragraph_text].append({
                    'text_to_remove': text_to_remove,
                    'original_item': item,
                    'filename': file_result.get('filename', 'unknown')
                })
        
        return paragraph_removals
    
    def clean_markdown_document(self, json_data: List[Dict], markdown_content: str, interactive: bool = True) -> str:
        """
        Clean the Markdown document based on strikethrough detection results.
        
        Args:
            json_data (List[Dict]): Strikethrough detection results
            markdown_content (str): Original Markdown content
            interactive (bool): Whether to ask for user confirmation before each removal
            
        Returns:
            str: Cleaned Markdown content
        """
        if not json_data:
            print("No JSON data to process.")
            return markdown_content
        
        # Group removals by paragraph to handle multiple removals in same paragraph
        paragraph_removals = self._group_removals_by_paragraph(json_data)
        
        cleaned_content = markdown_content
        total_removals = 0
        
        print("\nStarting document cleaning...")
        print("=" * 50)
        print(f"Found {len(paragraph_removals)} unique paragraphs with text to remove")
        
        for paragraph_idx, (paragraph_text, removals) in enumerate(paragraph_removals.items(), 1):
            print(f"\nProcessing paragraph {paragraph_idx}/{len(paragraph_removals)} with {len(removals)} removals")
            
            # Find the current paragraph in the document
            matches = self.find_paragraph_in_markdown(cleaned_content, paragraph_text)
            
            if not matches:
                if interactive:
                    print(f"\n{'='*60}")
                    print(f"WARNING: Could not find paragraph!")
                    print(f"{'='*60}")
                    print(f"Paragraph text:")
                    print(f"'{paragraph_text}'")
                    print(f"\nTexts to remove: {[r['text_to_remove'] for r in removals]}")
                    
                    while True:
                        response = input("\nHow to proceed? (s)kip this paragraph/(m)anual editing: ").lower().strip()
                        if response in ['s', 'skip']:
                            for removal in removals:
                                self.cleaning_log.append({
                                    'status': 'manual_skip',
                                    'message': f"User chose to skip - paragraph not found: '{paragraph_text[:50]}...'"
                                })
                            print(f"  - Skipped paragraph with {len(removals)} removals")
                            break
                        elif response in ['m', 'manual']:
                            # Manual editing mode
                            print(f"\n{'='*60}")
                            print("MANUAL EDITING MODE")
                            print(f"{'='*60}")
                            print("Enter the paragraph text as it appears in the markdown file:")
                            manual_paragraph = input("Paragraph text: ").strip()
                            
                            if not manual_paragraph:
                                print("No paragraph provided, skipping...")
                                for removal in removals:
                                    self.cleaning_log.append({
                                        'status': 'manual_skip',
                                        'message': f"User provided no paragraph, skipped: '{removal['text_to_remove']}'",
                                    })
                                print(f"  - Skipped paragraph with {len(removals)} removals")
                                break
                            
                            # Try to find the manual paragraph in markdown
                            manual_matches = self.find_paragraph_in_markdown(cleaned_content, manual_paragraph)
                            
                            if manual_matches:
                                manual_start, manual_end, matched_text = manual_matches[0]
                                current_paragraph = matched_text
                                
                                # Apply all removals to this paragraph
                                removals_applied = 0
                                for idx, removal in enumerate(removals, 1):
                                    text_to_remove = removal['text_to_remove']
                                    
                                    manual_text_to_remove = input(f"Enter text to remove {idx}/{len(removals)} (or press Enter to use: '{text_to_remove}'): ").strip()
                                    if not manual_text_to_remove:
                                        manual_text_to_remove = text_to_remove
                                    
                                    # Try removal with manual input
                                    updated_paragraph = self.remove_text_from_paragraph(current_paragraph, manual_text_to_remove)
                                    
                                    if updated_paragraph != current_paragraph:
                                        current_paragraph = updated_paragraph
                                        removals_applied += 1
                                        total_removals += 1
                                        
                                        self.cleaning_log.append({
                                            'status': 'manual_success',
                                            'message': f"Manually removed '{manual_text_to_remove}' from paragraph",
                                            'location': f"Paragraph {paragraph_idx}"
                                        })
                                        
                                        print(f"  ✓ Successfully removed: '{manual_text_to_remove[:30]}...'")
                                    else:
                                        self.cleaning_log.append({
                                            'status': 'manual_skip',
                                            'message': f"Manual removal failed: '{manual_text_to_remove}'"
                                        })
                                        print(f"  ⚠ Could not remove: '{manual_text_to_remove[:30]}...'")
                                
                                # Update the content with the manually cleaned paragraph
                                if current_paragraph != matched_text:
                                    cleaned_content = (
                                        cleaned_content[:manual_start] + 
                                        current_paragraph + 
                                        cleaned_content[manual_end:]
                                    )
                                    print(f"  ✓ Updated paragraph with {removals_applied} successful removals")
                                else:
                                    print(f"  - No changes made to paragraph")
                            else:
                                print(f"  ⚠ Could not find the manual paragraph in markdown file.")
                                for removal in removals:
                                    self.cleaning_log.append({
                                        'status': 'manual_skip',
                                        'message': f"Manual paragraph not found, skipped: '{removal['text_to_remove']}'"
                                    })
                                print(f"  - Skipped paragraph with {len(removals)} removals")
                            break
                        else:
                            print("Please enter 's' or 'm'")
                else:
                    for removal in removals:
                        self.cleaning_log.append({
                            'status': 'not_found',
                            'message': f"Could not find paragraph: '{paragraph_text[:50]}...'"
                        })
                    print(f"  ✗ Paragraph not found, skipping {len(removals)} removals")
                continue
            
            # Process all removals for this paragraph at once
            match_start, match_end, matched_paragraph = matches[0]
            current_paragraph = matched_paragraph
            
            # Apply all removals to this paragraph
            for idx, removal in enumerate(removals, 1):
                text_to_remove = removal['text_to_remove']
                
                if not text_to_remove:
                    self.cleaning_log.append({
                        'status': 'skipped',
                        'message': f"Removal {idx}: Missing text data"
                    })
                    continue
                
                print(f"Processing removal {idx}/{len(removals)}: '{text_to_remove[:30]}...'")
                
                # Remove text from the current paragraph
                updated_paragraph = self.remove_text_from_paragraph(current_paragraph, text_to_remove)
                
                if updated_paragraph != current_paragraph:
                    # Successful removal
                    current_paragraph = updated_paragraph
                    total_removals += 1
                    
                    self.cleaning_log.append({
                        'status': 'success',
                        'message': f"Removed '{text_to_remove}' from paragraph",
                        'location': f"Paragraph {paragraph_idx}"
                    })
                    
                    print(f"  ✓ Removed: '{text_to_remove[:30]}...'")
                else:
                    # Failed to remove - ask user if interactive mode is on
                    if interactive:
                        print(f"\n{'='*60}")
                        print(f"WARNING: Could not automatically remove text!")
                        print(f"{'='*60}")
                        print(f"Text to remove: '{text_to_remove}'")
                        print(f"From paragraph: '{current_paragraph[:100]}...'")
                        print(f"Reason: Text not found exactly in paragraph")
                        
                        while True:
                            response = input("\nHow to proceed? (s)kip this item/(m)anual editing: ").lower().strip()
                            if response in ['s', 'skip']:
                                self.cleaning_log.append({
                                    'status': 'manual_skip',
                                    'message': f"User chose to skip: '{text_to_remove}'",
                                    'location': f"Paragraph {paragraph_idx}"
                                })
                                print(f"  - Skipped: '{text_to_remove[:30]}...'")
                                break
                            elif response in ['m', 'manual']:
                                # Manual editing mode
                                print(f"\n{'='*60}")
                                print("MANUAL EDITING MODE")
                                print(f"{'='*60}")
                                print("Current paragraph:")
                                print(f"'{current_paragraph}'")
                                print(f"\nCurrent text to remove:")
                                print(f"'{text_to_remove}'")
                                
                                # Get manual input from user
                                print("\nPlease provide the corrected information:")
                                manual_text_to_remove = input("Enter the text to remove (or press Enter to use current): ").strip()
                                if not manual_text_to_remove:
                                    manual_text_to_remove = text_to_remove
                                
                                # Try removal with manual input
                                manual_cleaned = self.remove_text_from_paragraph(current_paragraph, manual_text_to_remove)
                                
                                if manual_cleaned != current_paragraph:
                                    # Successful manual removal
                                    current_paragraph = manual_cleaned
                                    total_removals += 1
                                    
                                    self.cleaning_log.append({
                                        'status': 'manual_success',
                                        'message': f"Manually removed '{manual_text_to_remove}' from paragraph",
                                        'location': f"Paragraph {paragraph_idx}"
                                    })
                                    
                                    print(f"  ✓ Successfully removed: '{manual_text_to_remove[:30]}...'")
                                    break
                                else:
                                    print(f"  ⚠ Manual removal also failed. The text '{manual_text_to_remove}' was not found in the paragraph.")
                                    print("Would you like to try again or skip this item?")
                                    retry_response = input("(r)etry manual editing/(s)kip: ").lower().strip()
                                    if retry_response in ['s', 'skip']:
                                        self.cleaning_log.append({
                                            'status': 'manual_skip',
                                            'message': f"Manual removal failed, user chose to skip: '{text_to_remove}'",
                                            'location': f"Paragraph {paragraph_idx}"
                                        })
                                        print(f"  - Skipped: '{text_to_remove[:30]}...'")
                                        break
                                    # If retry, continue the loop
                            else:
                                print("Please enter 's' or 'm'")
                    else:
                        # Non-interactive mode - log the failure and continue
                        self.cleaning_log.append({
                            'status': 'warning',
                            'message': f"Could not remove text: '{text_to_remove}'",
                            'location': f"Paragraph {paragraph_idx}"
                        })
                        print(f"  ⚠ Could not remove: '{text_to_remove[:30]}...'")
            
            # After processing all removals for this paragraph, update the content
            if current_paragraph != matched_paragraph:
                cleaned_content = (
                    cleaned_content[:match_start] + 
                    current_paragraph + 
                    cleaned_content[match_end:]
                )
                print(f"  ✓ Updated paragraph {paragraph_idx} in document")
        
        print(f"\nCleaning completed! Total removals: {total_removals}")
        return cleaned_content
    
    def save_cleaned_document(self, cleaned_content: str, output_path: str):
        """
        Save the cleaned Markdown document.
        
        Args:
            cleaned_content (str): The cleaned document content
            output_path (str): Path to save the cleaned document
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            print(f"\nCleaned document saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving cleaned document: {str(e)}")
    
    def print_cleaning_log(self):
        """Print the cleaning operation log."""
        print("\nCleaning Log:")
        print("=" * 50)
        
        if not self.cleaning_log:
            print("No log entries.")
            return
        
        status_counts = {}
        for entry in self.cleaning_log:
            status = entry['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
            status_symbol = {
                'success': '✓',
                'manual_success': '✓ (manual)',
                'warning': '⚠',
                'not_found': '✗',
                'skipped': '-',
                'manual_skip': '- (manual)'
            }.get(status, '?')
            
            print(f"{status_symbol} {entry['message']}")
            if 'location' in entry:
                print(f"  Location: {entry['location']}")
        
        print("\nSummary:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
    
    def clean_document(self, json_file_path: str, markdown_file_path: str, output_file_path: str, interactive: bool = True) -> bool:
        """
        Main function to clean a Markdown document based on JSON strikethrough data.
        
        Args:
            json_file_path (str): Path to JSON report
            markdown_file_path (str): Path to Markdown file to clean
            output_file_path (str): Path for cleaned output file
            interactive (bool): Whether to prompt user for each removal
            
        Returns:
            bool: True if cleaning was successful
        """
        # Load JSON data
        json_data = self.load_json_report(json_file_path)
        if not json_data:
            return False
        
        # Load Markdown content
        markdown_content = self.load_markdown_file(markdown_file_path)
        if not markdown_content:
            return False
        
        # Clean the document
        cleaned_content = self.clean_markdown_document(json_data, markdown_content, interactive)
        
        # Save cleaned document
        self.save_cleaned_document(cleaned_content, output_file_path)
        
        # Print log
        self.print_cleaning_log()
        
        return True


def main():
    """Main function to run the Markdown cleaner."""
    cleaner = MarkdownCleaner()
    
    try:
        success = cleaner.clean_document(JSON_FILE_PATH, MARKDOWN_FILE_PATH, OUTPUT_FILE_PATH)
        
        if success:
            print(f"\n{'='*50}")
            print("Document cleaning completed successfully!")
            print(f"Original file: {MARKDOWN_FILE_PATH}")
            print(f"Cleaned file: {OUTPUT_FILE_PATH}")
            print(f"JSON report: {JSON_FILE_PATH}")
        else:
            print("Document cleaning failed. Check the error messages above.")
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()