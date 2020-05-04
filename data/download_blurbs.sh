#!/bin/bash
if [ -f blurbs.txt ]; then
  echo "Nothing to do"
  exit 0
fi
if ! [ -f blurbs.zip ]; then
  echo "Downloading blurbs..."
  curl --output "blurbs.zip" "https://fiona.uni-hamburg.de/ca89b3cf/blurbgenrecollectionen.zip" 
fi
echo "Extracting zip..."
unzip blurbs.zip
echo "Reformatting data... (this takes some time)"
cat BlurbGenreCollection_EN_dev.txt BlurbGenreCollection_EN_test.txt BlurbGenreCollection_EN_train.txt | \
grep -E "title|body|isbn|d1" | \
sed -r "s/<\w+>([^<]*)<\/\w+>/\1/" | \
sed -r "s/<d1>/;/" | sed -r "s/<\/d1>//" > blurbs.txt
rm -r __MACOSX
rm BlurbGenreCollection_EN_dev.txt BlurbGenreCollection_EN_test.txt BlurbGenreCollection_EN_train.txt
