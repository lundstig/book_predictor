#!/bin/bash
if ! [ -f wiki-news-300d-1M-subword.vec ]; then
  if ! [ -f wiki-news-300d-1M-subword.vec.zip ]; then
    echo "Downloading embeddings..."
    curl --output "wiki-news-300d-1M-subword.vec.zip" "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip" 
  fi
  unzip -o "wiki-news-300d-1M-subword.vec.zip"
fi

if ! [ -f wiki-news-300d-1M-subword.bin ]; then
  if ! [ -f wiki-news-300d-1M-subword.bin.zip ]; then
    echo "Downloading subword information..."
    curl --output "wiki-news-300d-1M-subword.bin.zip" "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip" 
  fi
  unzip -o "wiki-news-300d-1M-subword.bin.zip"
fi
