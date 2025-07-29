"""
Module for reading and accessing the VOST dataset.
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

class Dataset:
    """
    Handles loading of the VOST dataset, including images and annotations.
    """
    SUBSET_OPTIONS: List[str] = ['train', 'val', 'test']
    VOID_LABEL: int = 255

    def __init__(self, root: Path, subset: str = 'val', sequences: Union[str, List[str]] = 'all'):
        """
        Initializes the Dataset object.
        """
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f"Subset must be one of {self.SUBSET_OPTIONS}, but got '{subset}'.")

        self.root: Path = root
        self.subset: str = subset
        self.img_path: Path = self.root / 'JPEGImages'
        self.mask_path: Path = self.root / 'Annotations'
        self.imagesets_path: Path = self.root / 'ImageSets'
        self._check_directories()

        sequence_names = self._get_sequence_names(sequences)
        self.sequences: Dict[str, Dict[str, List[str]]] = self._load_sequences(sequence_names)

    def _get_sequence_names(self, sequences: Union[str, List[str]]) -> List[str]:
        """Resolves sequence names from input."""
        if sequences == 'all':
            imageset_file = self.imagesets_path / f'{self.subset}.txt'
            with open(imageset_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return sequences if isinstance(sequences, list) else [sequences]

    def _load_sequences(self, sequence_names: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """Loads the file paths for images and masks for each sequence."""
        sequences_data = defaultdict(dict)
        for seq_name in sequence_names:
            mask_paths = sorted((self.mask_path / seq_name).glob('*.png'))
            if not mask_paths:
                raise FileNotFoundError(f"Annotations for sequence '{seq_name}' not found.")
            
            image_paths = sorted((self.img_path / seq_name).glob('*.jpg'))
            if not image_paths:
                raise FileNotFoundError(f"Images for sequence '{seq_name}' not found.")
            
            sequences_data[seq_name]['masks'] = [str(p) for p in mask_paths]
            sequences_data[seq_name]['images'] = [str(p) for p in image_paths]
        return sequences_data

    def _check_directories(self) -> None:
        """Verifies that necessary dataset directories exist."""
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found at: {self.root}")
        if not (self.imagesets_path / f'{self.subset}.txt').is_file():
            raise FileNotFoundError(f"ImageSet file for subset '{self.subset}' not found.")
        if self.subset in ['train', 'val'] and not self.mask_path.is_dir():
            raise FileNotFoundError("Annotations folder not found.")
    
    def get_all_masks(self, sequence: str, separate_objects: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Loads all masks for a given sequence, separating objects if requested.
        """
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        
        # Separate void pixels (value 255)
        void_mask = (masks == self.VOID_LABEL)
        masks[void_mask] = 0

        if separate_objects:
            num_objects = int(np.max(masks[0, ...]))
            if num_objects == 0:
                # Handle case with no objects by returning an empty dimension
                return np.zeros((0, *masks.shape)), void_mask, masks_id

            # Create a stack of masks, one for each object ID
            obj_ids = np.arange(1, num_objects + 1, dtype=masks.dtype)
            separated_masks = (masks[None, ...] == obj_ids[:, None, None, None])
            return separated_masks, void_mask, masks_id
            
        return masks, void_mask, masks_id
    
    def _get_all_elements(self, sequence: str, element_type: str) -> Tuple[np.ndarray, List[str]]:
        """A generic loader for images or masks for a sequence."""
        paths = self.sequences[sequence][element_type]
        if not paths:
            return np.array([]), []

        sample = np.array(Image.open(paths[0]))
        all_elements = np.zeros((len(paths), *sample.shape), dtype=sample.dtype)
        element_ids = []

        for i, path_str in enumerate(paths):
            path = Path(path_str)
            all_elements[i, ...] = np.array(Image.open(path))
            element_ids.append(path.stem)
            
        return all_elements, element_ids

    def get_sequences(self) -> Generator[str, None, None]:
        """Yields the name of each sequence in the dataset."""
        yield from self.sequences.keys()
