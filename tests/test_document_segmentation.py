import pytest
import numpy as np
import cv2
from src.preprocessing.document_segmentation import DocumentSegmenter

def test_document_segmenter_initialization():
    segmenter = DocumentSegmenter(target_width=800)
    assert segmenter.target_width == 800

def test_document_segmenter_blank_image():
    segmenter = DocumentSegmenter(target_width=800)
    blank_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    crops, vis = segmenter.segment_into_lines(blank_img)
    assert len(crops) == 0

def test_document_segmenter_synthetic_lines():
    segmenter = DocumentSegmenter(target_width=800)
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    
    # Draw simulated text lines (using thin strokes so adaptiveThreshold preserves them)
    for y in range(200, 240, 5):
        cv2.line(img, (100, y), (900, y), (0, 0, 0), 2)
    
    for y in range(400, 440, 5):
        cv2.line(img, (200, y), (800, y), (0, 0, 0), 2)
        
    # Add some vertical strokes so it's not detected as purely horizontal ruling lines
    for x in range(100, 900, 50):
        cv2.line(img, (x, 200), (x, 240), (0, 0, 0), 2)
    for x in range(200, 800, 50):
        cv2.line(img, (x, 400), (x, 440), (0, 0, 0), 2)
        
    crops, vis = segmenter.segment_into_lines(img)
    
    assert len(crops) == 2
    assert 30 <= crops[0].shape[0] <= 70
    assert 30 <= crops[1].shape[0] <= 70

def test_document_segmenter_ruled_background():
    segmenter = DocumentSegmenter(target_width=800)
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    
    # Ruled lines
    for y in range(100, 900, 100):
        cv2.line(img, (0, y), (1000, y), (200, 200, 200), 2)
        
    # Text block
    for y in range(300, 350, 5):
        cv2.line(img, (100, y), (900, y), (0, 0, 0), 2)
    for x in range(100, 900, 30):
        cv2.line(img, (x, 300), (x, 350), (0, 0, 0), 2)
        
    crops, vis = segmenter.segment_into_lines(img)
    assert len(crops) == 1

def test_document_segmenter_deskew():
    segmenter = DocumentSegmenter(target_width=800)
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    
    # Draw text block
    for y in range(480, 520, 5):
        cv2.line(img, (200, y), (800, y), (0, 0, 0), 2)
    for x in range(200, 800, 30):
        cv2.line(img, (x, 480), (x, 520), (0, 0, 0), 2)
        
    # Rotate
    center = (500, 500)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)
    rotated_img = cv2.warpAffine(img, M, (1000, 1000), borderValue=(255, 255, 255))
    
    crops, vis = segmenter.segment_into_lines(rotated_img)
    assert len(crops) == 1
    assert 30 <= crops[0].shape[0] <= 90

def test_document_segmenter_touching_lines():
    segmenter = DocumentSegmenter(target_width=800)
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    
    # Line 1
    for y in range(100, 180, 5):
        cv2.line(img, (100, y), (900, y), (0, 0, 0), 2)
    for x in range(100, 900, 30):
        cv2.line(img, (x, 100), (x, 180), (0, 0, 0), 2)
        
    # Line 2 (touching at a few points)
    for y in range(190, 270, 5):
        cv2.line(img, (100, y), (900, y), (0, 0, 0), 2)
    for x in range(100, 900, 30):
        cv2.line(img, (x, 190), (x, 270), (0, 0, 0), 2)
        
    # Valley connectors
    cv2.line(img, (300, 180), (300, 190), (0, 0, 0), 2)
    cv2.line(img, (600, 180), (600, 190), (0, 0, 0), 2)
    
    crops, vis = segmenter.segment_into_lines(img)
    assert len(crops) >= 2

