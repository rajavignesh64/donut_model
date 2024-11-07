
import os
import time
import json
import re
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import pandas as pd
from datetime import datetime

# Load pre-trained Donut model and processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def convert_none_values(data):
    """Convert None values to 'None' string in a dictionary or list."""
    if isinstance(data, dict):
        # Iterate over each key-value pair in the dictionary
        return {key: 'None' if value is None else value for key, value in data.items()}
    elif isinstance(data, list):
        # If the data is a list, check each item
        return [convert_none_values(item) for item in data]
    else:
        # Return the data as is if it's neither None, a dictionary, nor a list
        return data

def validate_data(data):
    # Validate company name (should be alphabetic)
    def is_valid_company_name(company_name):
        return company_name.isalpha()

    # Validate date (should be in dd/mm/yyyy format)
    def is_valid_date(date):
        try:
            datetime.strptime(date, '%d/%m/%Y')
            return True
        except ValueError:
            return False

    # Validate address (should be alphanumeric and spaces, can contain other punctuation)
    def is_valid_address(address):
        return bool(re.match(r'^[a-zA-Z0-9\s,.-]*$', address))

    # Validate total amount (should be int or float)
    def is_valid_total_amount(amount):
        try:
            float_amount = float(amount)
            return isinstance(float_amount, (int, float))
        except ValueError:
            return False

    # Validate all fields
    is_company_name_valid = is_valid_company_name(data['company_name'])
    is_date_valid = is_valid_date(data['date'])
    is_address_valid = is_valid_address(data['address'])
    is_total_amount_valid = is_valid_total_amount(data['total_amount'])

    # Count valid fields
    valid_fields_count = sum([is_company_name_valid, is_date_valid, is_address_valid, is_total_amount_valid])
    
    # Calculate percentage accuracy
    accuracy_percentage = (valid_fields_count / 4) * 100  # There are 4 fields to validate
    
    # Return the accuracy percentage
    return accuracy_percentage


def extract_first_date(data):
    # Regular expression pattern to match date format: dd/mm/yyyy or dd-mm-yyyy
    date_pattern = r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b'
    
    # Helper function to recursively search for dates in nested structures
    def find_dates_in_structure(structure):
        if isinstance(structure, dict):
            # If it's a dictionary, check each key-value pair
            for key, value in structure.items():
                # Recursively search in the value
                result = find_dates_in_structure(value)
                if result:
                    return result
        elif isinstance(structure, list):
            # If it's a list, check each item
            for item in structure:
                result = find_dates_in_structure(item)
                if result:
                    return result
        elif isinstance(structure, str):
            # If it's a string, search for a date using regex
            match = re.search(date_pattern, structure)
            if match:
                return match.group(1)
        return None
    
    # Start the recursive search from the top level of the structure
    return find_dates_in_structure(data)


def extract_company_name(data):

    data=data[0]
    """Extract the company name from the dict"""
    nm_values = []  # List to store the 'nm' values
    found_nm_in_current_dict = False  # Flag to track if we've found 'nm' in the current dictionary
    def recursive_search(obj):
        nonlocal found_nm_in_current_dict
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'nm' and not found_nm_in_current_dict:
                    # If the 'nm' key is found for the first time, extract and stop further searching for 'nm' in this dict
                    nm_values.append(value)
                    found_nm_in_current_dict = True  # Mark that we've found 'nm' in this dictionary
                elif isinstance(value, (dict, list)):
                    # Continue recursively searching if the value is a dictionary or list
                    recursive_search(value)
        elif isinstance(obj, list):
            # Iterate over the list and recursively search
            for item in obj:
                recursive_search(item)
    # Start the recursive search
    recursive_search(data)

    return nm_values



def extract_address(data):
    address = []
    nm_found = False  # Flag to track if the first "nm" has been found

    def recursive_search(obj):
        nonlocal nm_found
        
        if isinstance(obj, dict):
            # Iterate over dictionary keys and values
            for key, value in obj.items():
                if key == "nm" and not nm_found:
                    # If the first "nm" is found, just flag it and skip concatenation
                    nm_found = True
                elif key == "nm":
                    # If "nm" is found after the first, concatenate its value
                    address.append(str(value))
                # Recursively check the value
                recursive_search(value)
        
        elif isinstance(obj, list):
            # If the object is a list, iterate over all items in the list
            for item in obj:
                recursive_search(item)
        
        elif isinstance(obj, str):
            # If it's a string, it could be part of an address or any other info
            return  # Strings are handled by the key "nm" check
        
        # Handle other data types (int, float, etc.) by just returning
        # and skipping them if they don't have any relevant "nm" field.
        else:
            return

    # Start recursive search
    recursive_search(data)

    # Concatenate the address components into a single string
    return " ".join(address) if address else "Address not found"


def extract_total_or_price(data):
    """Extract the first occurrence of 'total_price' or 'price' (as a number), return 0 if not found."""
    total_price = 0  # Default value if neither is found

    def recursive_search(obj):
        nonlocal total_price

        if isinstance(obj, dict):
            # Check for 'total_price' first
            if 'total_price' in obj and isinstance(obj['total_price'], (int, float, str)):
                try:
                    # Attempt to convert to a numeric value (in case it's a string)
                    total_price = float(obj['total_price'].replace(',', ''))  # Handle potential commas in numbers
                except ValueError:
                    total_price = 0
                return  # Stop further recursion if 'total_price' is found
            
            # Check for 'price' if 'total_price' is not found
            elif 'price' in obj:
                price_value = obj['price']
                # If 'price' is a number (int or float), take its value
                if isinstance(price_value, (int, float)):
                    total_price = price_value
                    return  # Stop further recursion if a valid number is found
                elif isinstance(price_value, str):
                    try:
                        # If 'price' is a string, attempt to convert it to a float
                        total_price = float(price_value.replace(',', ''))  # Handle potential commas in numbers
                        return
                    except ValueError:
                        total_price = 0
                        return

            # If the object is a dictionary and neither 'total_price' nor 'price' is found
            for key, value in obj.items():
                if isinstance(value, (dict, list)):  # Recurse further into nested structures
                    recursive_search(value)

        elif isinstance(obj, list):
            for item in obj:
                recursive_search(item)

    # Start searching the data
    recursive_search(data)
    return total_price



def get_format_data(extracted_data):
    """Process the receipt and extract the required fields."""
    
    # Handle cases where 'menu' can be either a list or dict
    company_name = extract_company_name(extracted_data)

    if isinstance(company_name,list):
        company_name=flatten_list(company_name)

    print('company_name:',''.join(company_name))

    company_name=''.join(company_name)

    date = extract_first_date(extracted_data)
    print('date:',date)
    address = extract_address(extracted_data)

    print('address:',address)


    total_amount = extract_total_or_price(extracted_data)

    print('total_amount:',total_amount)

    # # Format the extracted data in JSON
    extracted_fields = {
        "company_name": company_name,
        "date": date,
        "address": address,
        "total_amount": total_amount
    }

    return extracted_fields


def flatten_list(nested_list):
    """Recursively flattens a nested list into a single list."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            # If the item is a list, recursively flatten it
            flat_list.extend(flatten_list(item))
        else:
            # Otherwise, add the item directly
            flat_list.append(item)
    return flat_list



# Function to process an image and extract structured data
def extract_data_from_image(image_path):
    image = Image.open(image_path)

    # Prepare the task prompt and decoder input
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # Process the image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Generate the outputs from the model
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decode the output sequence
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove task start token

    # Process the sequence to extract relevant fields (company_name, date, address, total_amount)
    extracted_data = processor.token2json(sequence)
    
    # print('extracted_data:',extracted_data)
    return extracted_data



# Function to process all images in a directory
def process_receipts(image_directory):
    
    accuracy_list = []
    time_list = []
    image_name_lis=[]
    accuracy_lis=[]
    processing_time_lis=[]

    for image_name in os.listdir(image_directory):
        if image_name.endswith((".jpg", ".png", ".jpeg")):

            image_path = os.path.join(image_directory, image_name)

            start_time = time.time()
            extracted_data = extract_data_from_image(image_path)
            end_time = time.time()

            processing_time = end_time - start_time
            results = []

            extracted_data=[{key: value} for key, value in extracted_data.items()]

            # Extract fields
            extracted_fields = get_format_data(extracted_data)

            print(f'******EXTRACTED JSON FOR {image_name}***********')
            print(extracted_fields)
            print('*************************************************')
            extracted_fields=convert_none_values(extracted_fields)
            results.append(extracted_fields)

            accuracy = validate_data(extracted_fields)

            print(f'***********REPORT FOR {image_name}***********')
            print('image_name:',image_name)        
            print('accuracy:',accuracy)
            print('processing_time:',processing_time)
            print('***********************************************')


            image_name_lis.append(image_name)
            accuracy_lis.append(accuracy)
            processing_time_lis.append(processing_time)

            outputfolder = r'..\output\\'

            output_file_path = os.path.join(outputfolder, f"{image_name.split('.')[0]}.json")

            # Save results to a JSON file
            with open(output_file_path, "w") as f:
                json.dump(results, f, indent=4)

    report_dict={'image_name':image_name_lis,'accuracy':accuracy_lis,
                 'processing_time':processing_time_lis}


    df=pd.DataFrame(report_dict)

    report_path=r'..\report\\'

    report_file_path = os.path.join(report_path, 'detailed_report_updated.csv')
    df.to_csv(report_file_path,index=False)

    print('**********************************************************************************************')
    print('Process has been Sucessfully Completed get the output files in output directory***************')
    print('**********************************************************************************************')

    
if __name__=='__main__':
    # Run the process
    image_directory=r'..\source_image'

    process_receipts(image_directory)
