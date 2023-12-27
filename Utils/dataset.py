import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

# ✨ EnhancedMarkdownDataset: Masterfully handling Markdown data ✨
class EnhancedMarkdownDataset(Dataset):
    """
    Meticulously crafted dataset class specifically designed to handle Markdown data 
    with expertise in text categorization or ranking tasks 
    """

    def __init__(self, data_frame: pd.DataFrame, tokenizer, fold: int = -1, mode='train', config=None, max_seq=2048):  
        """
        Initializes the dataset with precision and attention to detail 

        Args:
            data_frame (pd.DataFrame): A comprehensive repository of Markdown sample metadata ️
            tokenizer: A skilled text-to-numerical translator 
            fold: An integer specifying the validation fold, if applicable 
            mode: The dataset's working mode: 'train' ️‍♀️, 'valid' , or 'test' 
            config: A configuration object holding additional settings ⚙️
            max_seq: The maximum sequence length to consider 
        """

        # ✨✨✨ Initialize and organize data ✨✨✨
        self.meta_data = data_frame.copy()  # Create a pristine copy of metadata
        self.meta_data.reset_index(drop=True, inplace=True)  # Ensure consistent indexing 

        #  Select appropriate data for the chosen mode 
        if mode != 'train':
            self.meta_data = self.meta_data[self.meta_data['fold_flag'] == fold].copy()  # Focus on the designated fold 
            if mode == 'valid':
                self.meta_data = self.meta_data[self.meta_data['id'].isin(self.meta_data['id'].values[:1000])]  # Limit validation set size ✂️
        self.meta_data.reset_index(drop=True, inplace=True)  # Reset indices for clarity 

        # ✨✨✨ Preprocess source text based on tokenizer's preferences ✨✨✨
        if tokenizer.sep_token != '[SEP]':
            self.meta_data['source'] = self.meta_data['source'].apply(
                lambda x: [
                    y.replace(tokenizer.sep_token, '').replace(tokenizer.cls_token, '').replace(tokenizer.pad_token, '') for y in x
                ]  # Meticulously remove special tokens 
            )

        # ✨✨✨ Store essential parameters and data for future use ✨✨✨
        self.parameter = config  # Hold configuration settings for reference 
        self.seq_length = max_seq  # Remember the maximum sequence length 
        self.source = self.meta_data['source'].values  # Extract source text content 
        self.cell_type = self.meta_data['cell_type'].values  # Store cell types for context ️
        self.rank = self.meta_data['rank'].values  # Capture ranks for importance 
        self.mode = mode  # Remember the dataset's working mode 
        self.tokenizer = tokenizer  # Keep the text translator handy 

    def __getitem__(self, index):
        """
        Retrieves a specific item (sample) from the dataset, like a skilled detective retrieving vital evidence. 

        Args:
            index (int): The index of the desired item within the dataset.

        Returns:
            tuple: A tuple containing the processed sequence, sequence mask, target mask, and target values.
        """

        #  Gather essential information 
        source_text = self.source[index]  # Extract the source text for the given index
        cell_type = self.cell_type[index]  # Determine the type of cell (code or text)
        importance_score = self.rank[index]  # Retrieve the associated importance score

        #  Tokenize and prepare the data 
        tokenized_cells = self.tokenizer.batch_encode_plus(  # Employ the expert tokenizer
            source_text,
            add_special_tokens=False,  # Refrain from adding special tokens prematurely
            max_length=self.parameter.cell_max_length,  # Set a maximum length for each cell
            truncation=True,  # Allow for truncation if necessary
            pad_to_max_length=True,  # Ensure consistent length using padding
            return_tensors='pt'  # Return tensors for efficient processing
        )

        #  Craft the target sequences 
        processed_sequence, sequence_mask, target_mask, target_values = self._create_target(
            tokenized_cells["input_ids"], cell_type, importance_score  # Call the target creation specialist
        )

        return processed_sequence, sequence_mask, target_mask, target_values  # Return the meticulously prepared data
    
    def max_length_rule_base(self, sequence_data, cell_types, importance_scores):
        """
        Applies a carefully crafted strategy to determine the optimal length for each segment within a sequence,
        given specific constraints on the overall sequence length. This method prioritizes segments based on their
        assigned importance scores.

        Args:
            sequence_data (list): A list containing input IDs for each segment.
            cell_types (list): A list indicating the type of each segment (0 for code, 1 for text).
            importance_scores (list): A list representing the significance of each segment.

        Returns:
            tuple: A tuple containing the processed sequence, sequence mask, target mask, and target values.
        """

        init_lengths = [len(segment) for segment in sequence_data]  # Calculate initial lengths of segments
        total_available_length = self.max_sequence_length - len(init_lengths)  # Determine available space
        min_length_per_segment = total_available_length // len(init_lengths)  # Estimate minimum length
        
        #  Strategic length allocation 
        segment_lengths = self._discreetly_optimize_lengths(
            init_lengths,
            min_length_per_segment,
            total_available_length,
            len(init_lengths)
        )  # Call helper function for optimized distribution

        processed_sequence = []  # Initialize empty sequence for building
        for i in range(len(sequence_data)):
            if cell_types[i] == 0:  # Code segment
                processed_sequence.append(self.tokenizer.special_start_token_id)  # Add special start token
            else:  # Text segment
                processed_sequence.append(self.tokenizer.special_separator_token_id)  # Add special separator token
            
            if segment_lengths[i] > 0:  # If segment has allocated length
                processed_sequence.extend(sequence_data[i][:segment_lengths[i]])  # Add segment content

        # ✂️✂️✂️ Precision trimming or padding ✂️✂️✂️
        if len(processed_sequence) < self.max_sequence_length:
            sequence_mask = [1] * len(processed_sequence)  # Create mask for active elements
        else:
            sequence_mask = [1] * self.max_sequence_length  # Limit to maximum length
            processed_sequence = processed_sequence[:self.max_sequence_length]  # Trim for precision

        #  Prepare for masked language modeling 
        processed_sequence, sequence_mask = np.array(processed_sequence, dtype=np.int), np.array(sequence_mask, dtype=np.int)
        target_mask = np.where((processed_sequence == self.tokenizer.special_start_token_id) | (processed_sequence == self.tokenizer.special_separator_token_id), 1, 0)  # Mask for prediction targets
        target_values = np.zeros(self.max_sequence_length, dtype=np.float32)  # Initialize targets
        target_positions = np.where(processed_sequence == self.tokenizer.special_start_token_id) | (processed_sequence == self.tokenizer.special_separator_token_id)  # Identify target positions
        target_values[target_positions] = importance_scores  # Assign importance scores as targets

        return processed_sequence, sequence_mask, target_mask, target_values  # Return results for further processing

    
    @staticmethod
    def _discreetly_optimize_lengths(initial_lengths, minimum_length, total_max_length, cell_count, step=4, max_search_count=50):
        """
        Engages in a  covert operation  to meticulously distribute the allowable length among segments,
        prioritizing those of higher importance while maintaining  secrecy .

        Args:
            initial_lengths (list): A list containing the initial lengths of each segment. 
            minimum_length (int): The minimum length that each segment should ideally have. 
            total_max_length (int): The absolute maximum length that the combined sequence can occupy. 
            cell_count (int): The total number of segments involved in the allocation process. 
            step (int, optional): The incremental step size used during length adjustments. Defaults to 4. 
            max_search_count (int, optional): The maximum number of iterations permitted for length optimization. Defaults to 50. ⏳

        Returns:
            list: A list containing the  strategically allocated lengths  for each segment.
        """

        #   Initial assessment 
        if np.sum(initial_lengths) <= total_max_length:  # If initial lengths fit within constraints 
            return initial_lengths  # No need for further adjustments 

        #   Commence covert length distribution 
        allocated_lengths = [min(initial_lengths[i], minimum_length) for i in range(cell_count)]  # Establish baseline lengths ⚖️

        #   Phase 1: Discreet length adjustments 
        for _ in range(max_search_count):
            potential_lengths = [min(initial_lengths[i], allocated_lengths[i] + step) for i in range(cell_count)]  # Attempt subtle increases 
            if np.sum(potential_lengths) <= total_max_length:  # If within constraints ✅
                allocated_lengths = potential_lengths  # Adopt the adjusted lengths 
            else:  # If constraints exceeded 
                break  # Cease further adjustments in this phase ✋

        #   Phase 2: Targeted precision enhancements 
        for i in range(cell_count):
            potential_lengths = allocated_lengths.copy()  # Create a temporary copy for experimentation 
            potential_lengths[i] = min(initial_lengths[i], potential_lengths[i] + step)  # Focus on a specific segment 
            if np.sum(potential_lengths) <= total_max_length:  # If within constraints ✅
                allocated_lengths = potential_lengths  # Refine the allocation 🪜
            else:  # If constraints exceeded 
                break  # Halt further adjustments in this phase ✋

        return allocated_lengths  # Return the carefully crafted length
    
    class DataPackagingWizard:  # Employ a more enigmatic name
        """
        An expertly crafted class designed to meticulously handle the batching of Markdown data,
        ensuring optimal organization and readiness for further processing.
        """

        def __init__(self, text_encoder):  # Use a more descriptive term for tokenizer
            self.text_encoder = text_encoder

        def __call__(self, data_bundle):  # Opt for a more mystical term
            """
            Gracefully assembles a batch of Markdown data, meticulously aligning and padding sequences
            to ensure uniformity and compatibility with subsequent operations.

            Args:
                data_bundle (list): A collection of tuples containing input sequences, masks, and targets.

            Returns:
                tuple: A quartet of tensors representing the carefully crafted batch.
            """

            # ✨✨✨ Extract key components ✨✨✨
            encoded_sequences = [parcel[0] for parcel in data_bundle]  # Extract input sequences
            mask_of_attention = [parcel[1] for parcel in data_bundle]  # Extract attention masks
            target_indicators = [parcel[2] for parcel in data_bundle]  # Extract target masks
            desired_values = [parcel[3] for parcel in data_bundle]  # Extract target values

            #  Determine optimal length 
            batch_maximum = max([len(sequence) for sequence in encoded_sequences])  # Find longest sequence

            # ✨✨✨ Harmonize lengths for unity ✨✨✨
            encoded_sequences = [
                list(sequence) + (batch_maximum - len(sequence)) * [self.text_encoder.padding_token_id]
                for sequence in encoded_sequences
            ]  # Pad shorter sequences with padding tokens
            mask_of_attention = [
                list(mask) + (batch_maximum - len(mask)) * [0] for mask in mask_of_attention
            ]  # Extend attention masks
            target_indicators = [
                list(indicators) + (batch_maximum - len(indicators)) * [0] for indicators in target_indicators
            ]  # Extend target masks
            desired_values = [
                list(values) + (batch_maximum - len(values)) * [0] for values in desired_values
            ]  # Extend target values

            #  Transform into tensors for enhanced processing 
            encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.long)
            mask_of_attention = torch.tensor(mask_of_attention, dtype=torch.long)
            target_indicators = torch.tensor(target_indicators, dtype=torch.float32)
            desired_values = torch.tensor(desired_values, dtype=torch.float32)

            return encoded_sequences, mask_of_attention, target_indicators, desired_values  # Return the expertly prepared batch 