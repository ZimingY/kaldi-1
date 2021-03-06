// feat/wave-reader.cc

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "feat/wave-reader.h"
#include "base/kaldi-error.h"
#include "base/kaldi-utils.h"

#include <cstdio>
#include <sstream>

namespace kaldi {


// static
void WaveData::Expect4ByteTag(std::istream &is, const char *expected) {
  char tmp[5];
  tmp[4] = '\0';
  is.read(tmp, 4);
  if (is.fail()) KALDI_ERR << "WaveData: expected " << expected << ", failed to read anything";
  if (strcmp(tmp, expected))
    KALDI_ERR << "WaveData: expected " << expected << ", got " << tmp;
}

// static
uint32 WaveData::ReadUint32(std::istream &is) {
  char result[4];
  is.read(result, 4);
#ifdef __BIG_ENDIAN__
  KALDI_SWAP4(result);
#endif
  if (is.fail())
    KALDI_ERR << "WaveData: unexpected end of file.";
  return * reinterpret_cast<uint32*>(result);
}

// static
uint16 WaveData::ReadUint16(std::istream &is) {
  char result[2];
  is.read(result, 2);
#ifdef __BIG_ENDIAN__
  KALDI_SWAP2(result);
#endif
  if (is.fail())
    KALDI_ERR << "WaveData: unexpected end of file.";
  return * reinterpret_cast<uint16*>(result);
}

// static
void WaveData::Read4ByteTag(std::istream &is, char *dest) {
  is.read(dest, 4);
  if (is.fail())
    KALDI_ERR << "WaveData: expected 4-byte chunk-name, got read errror";
}

// static
void WaveData::WriteUint32(std::ostream &os, int32 i) {
  char buf[4];
  *(reinterpret_cast<uint32*>(buf)) = i;
#ifdef __BIG_ENDIAN__
  KALDI_SWAP4(result);
#endif
  os.write(buf, 4);
  if(os.fail())
    KALDI_ERR << "WaveData: error writing to stream.";
}

void WaveData::WriteUint16(std::ostream &os, int16 i) {
  char buf[4];
  *(reinterpret_cast<uint16*>(buf)) = i;
#ifdef __BIG_ENDIAN__
  KALDI_SWAP2(result);
#endif
  os.write(buf, 2);
  if(os.fail())
    KALDI_ERR << "WaveData: error writing to stream.";
}



void WaveData::Read(std::istream &is) {
  data_.Resize(0, 0);  // clear the data.

  Expect4ByteTag(is, "RIFF");
  uint32 riff_chunk_size = ReadUint32(is);
  Expect4ByteTag(is, "WAVE");

  uint32 riff_chunk_read = 0;
  riff_chunk_read += 4;  // WAVE included in riff_chunk_size.

  Expect4ByteTag(is, "fmt ");
  uint32 subchunk1_size = ReadUint32(is);
  uint16 audio_format = ReadUint16(is),
      num_channels = ReadUint16(is);
  uint32 sample_rate = ReadUint32(is),
      byte_rate = ReadUint32(is),
      block_align = ReadUint16(is),
      bits_per_sample = ReadUint16(is);

  if (audio_format != 1)
    KALDI_ERR << "WaveData: can read only PCM data, audio_format is not 1: "
              << audio_format;
  if (subchunk1_size < 16)
    KALDI_ERR << "WaveData: expect PCM format data to have fmt chunk of at least size 16.";
  else
    for (uint32 i = 16; i < subchunk1_size; i++) is.get();  // use up extra data.

  if (num_channels <= 0)
    KALDI_ERR << "WaveData: no channels present";
  samp_freq_ = static_cast<BaseFloat>(sample_rate);
  if (bits_per_sample != 8 && bits_per_sample != 16 && bits_per_sample != 32)
    KALDI_ERR << "WaveData: bits_per_sample is " << bits_per_sample;
  if (byte_rate != sample_rate * bits_per_sample/8 * num_channels)
    KALDI_ERR << "Unexpected byte rate " << byte_rate << " vs. "
              << sample_rate <<" * " << (bits_per_sample/8)
              << " * " << num_channels;
  if (block_align != num_channels * bits_per_sample/8)
    KALDI_ERR << "Unexpected block_align: " << block_align << " vs. "
              << num_channels << " * " << (bits_per_sample/8);

  riff_chunk_read += 8 + subchunk1_size;
  // size of what we just read, 4 bytes for "fmt " + 4
  // for subchunk1_size + subchunk1_size itself.

  // We support an optional "fact" chunk (which is useless but which
  // we encountered), and then a single "data" chunk.

  char next_chunk_name[4];
  Read4ByteTag(is, next_chunk_name);
  riff_chunk_read += 4;

  if (!strncmp(next_chunk_name, "fact", 4)) {
    // will just ignore the "fact" chunk.  for non-compressed data
    // (which we don't support anyway), it doesn't contain useful information.
    uint32 chunk_sz = ReadUint32(is);
    if (chunk_sz != 4)
      KALDI_WARN << "Expected fact chunk to be 4 bytes long.";
    for (uint32 i = 0; i < chunk_sz; i++)
      is.get();
    riff_chunk_read += 4 + chunk_sz;  // for chunk_sz (4) + chunk contents (chunk-sz)

    // Now read the next chunk name.
    Read4ByteTag(is, next_chunk_name);
    riff_chunk_read += 4;
  }

  if (strncmp(next_chunk_name, "data", 4))
    KALDI_ERR << "WaveData: expected data chunk, got instead "
              << next_chunk_name;

  uint32 data_chunk_size = ReadUint32(is);
  riff_chunk_read += 4;
  std::vector<char> chunk_data_vec(data_chunk_size);
  char *data_ptr = &(chunk_data_vec[0]);
  is.read(data_ptr, data_chunk_size);
  riff_chunk_read += data_chunk_size;

  if (riff_chunk_read  != riff_chunk_size)
    KALDI_WARN << "Expected " << riff_chunk_size << " bytes in RIFF chunk, but got "
               << riff_chunk_read << " (do not support reading multiple data chunks).";
  if (is.fail())
    KALDI_ERR << "WaveData: failed to read data chunk.";

  if (data_chunk_size % block_align != 0)
    KALDI_ERR << "WaveData: data chunk size has unexpected length "
              << data_chunk_size << "; block-align = " << block_align;
  if (data_chunk_size == 0)
    KALDI_ERR << "WaveData: empty file (no data)";

  uint32 num_samp = data_chunk_size / block_align;
  data_.Resize(num_channels, num_samp);
  for (uint32 i = 0; i < num_samp; i++) {
    for (uint32 j = 0; j < num_channels; j++) {
      switch (bits_per_sample) {
        case 8:
          data_(j, i) = *data_ptr;
          data_ptr++;
          break;
        case 16:
          {
            int16 k = *reinterpret_cast<uint16*>(data_ptr);
#ifdef __BIG_ENDIAN__
            KALDI_SWAP2(k);
#endif
            data_(j, i) =  k;
            data_ptr += 2;
            break;
          }
        case 32:
          {
            int32 k = *reinterpret_cast<uint32*>(data_ptr);
#ifdef __BIG_ENDIAN__
            KALDI_SWAP4(k);
#endif
            data_(j, i) =  k;
            data_ptr += 4;
            break;
          }
        default:
          KALDI_ERR << "bits per sample is " << bits_per_sample;  // already checked this.
      }
    }
  }
}


// Write 16-bit PCM.

// note: the WAVE chunk contains 2 subchunks.
//
// subchunk2size = data.NumRows() * data.NumCols() * 2.


void WaveData::Write(std::ostream &os) const {
  os << "RIFF";
  if(data_.NumRows() == 0)
    KALDI_ERR << "Error: attempting to write empty WAVE file";
  
  int32 num_chan = data_.NumRows(),
      num_samp = data_.NumCols(),
      bytes_per_samp = 2;

  int32 subchunk2size = (num_chan * num_samp * bytes_per_samp);
  int32 chunk_size = 36 + subchunk2size;
  WriteUint32(os, chunk_size);
  os << "WAVE";
  os << "fmt ";
  WriteUint32(os, 16);
  WriteUint16(os, 1);
  WriteUint16(os, num_chan);
  KALDI_ASSERT(samp_freq_ > 0);
  WriteUint32(os, static_cast<int32>(samp_freq_));
  WriteUint32(os, static_cast<int32>(samp_freq_) * num_chan * bytes_per_samp);
  WriteUint16(os, num_chan * bytes_per_samp);
  WriteUint16(os, 8 * bytes_per_samp);
  os << "data";
  WriteUint32(os, subchunk2size);

  const BaseFloat *data_ptr = data_.Data();
  int32 stride = data_.Stride();
      
  for(int32 i = 0; i < num_samp; i++) {
    for(int32 j = 0; j < num_chan; j++) {
      int32 elem = static_cast<int32>(data_ptr[j*stride + i]);
      int16 elem_16(elem);
      if(static_cast<int32>(elem_16) != elem)
        KALDI_ERR << "Wave file is out of range for 16-bit.";
#ifdef __BIG_ENDIAN__
      KALDI_SWAP2(elem_16);
#endif
      os.write(reinterpret_cast<char*>(&elem_16), 2);
    }
  }
  if(os.fail())
    KALDI_ERR << "Error writing wave data to stream.";
}


} // end namespace
