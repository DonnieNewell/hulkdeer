#include "Decomposition.h"
#include <stdio.h>
#include <algorithm>
#include <limits>
#include <boost/scoped_ptr.hpp>

Decomposition::Decomposition() {
}

Decomposition::Decomposition(const Decomposition &d) {
  *this = d;
}

Decomposition::~Decomposition() {
  for (size_t s = 0; s < domain.size(); ++s) {
    if (NULL != domain.at(s)) {
      delete domain.at(s);
      domain.at(s) = NULL;
    }
  }
  domain.clear();
}

Decomposition& Decomposition::operator=(const Decomposition& d) {
  this->domain.clear();
  for (size_t i = 0; i < d.getNumSubDomains(); ++i) {
    SubDomain* s = new SubDomain(*(d.getSubDomain(i)));
    this->domain.push_back(s);
  }
  return *this;
}

const SubDomain* Decomposition::getSubDomain(const int index) const {
  return this->domain.at(index);
}

SubDomain* Decomposition::popSubDomain() {
  SubDomain* tmp = this->domain.back();
  this->domain.pop_back();
  return tmp;
}

bool zOffsetCompare(SubDomain* first, SubDomain* second) {
  return first->getOffset(0) < second->getOffset(0);
}

/**
 * 
 * @param kNumBlocks number of blocks to combine into one block
 * @return pointer to new aggregate block
 */
SubDomain* Decomposition::getAggregate3D(const unsigned int kNumBlocksToCombine) {
  const unsigned int kFirstBlockIndex = 0;
  unsigned int num_elements_z = 0;
  int fake_neighbors[26] = {0};
  // calculate how deep the new block will be by
  //   taking overlapped GZ's into account.
  SubDomain* new_subdomain = NULL;
  if (0 == this->domain.size()) return NULL;
  
  if (1 == kNumBlocksToCombine) {
    new_subdomain = domain.front();
    domain.pop_front();
    return new_subdomain;
  }
  const int kBorderZ = domain.front()->getBorder(0);
  int max_z_offset = 0,
          min_z_offset = numeric_limits<int>::max(),
          max_z_length = 0;
  for (unsigned int i = 0; i < kNumBlocksToCombine; ++i) {
    int old_max = max_z_offset;
    max_z_offset = std::max(max_z_offset, domain.at(i)->getOffset(0));
    max_z_length = (max_z_offset > old_max) ?
                    domain.at(i)->getLength(0) :
                    max_z_length;
    min_z_offset = std::min(min_z_offset, domain.at(i)->getOffset(0));
  }
  new_subdomain = domain.at(kFirstBlockIndex);
  num_elements_z = new_subdomain->getLength(0) - 2 * kBorderZ;
  num_elements_z *= kNumBlocksToCombine;
  num_elements_z += 2 * kBorderZ;
  int* id = new_subdomain->getId(); // use id of first block 

  // because we are combining blocks, we have to set id of next available block
  // to be consistent post-aggregation.
  if (kNumBlocksToCombine + 1 <= domain.size()) {
    const int* kNextId = domain.at(1)->getId();
    domain.at(kNumBlocksToCombine)->setId(kNextId[0], kNextId[1], kNextId[2]);
  }
  const int kZOffset = new_subdomain->getOffset(0);
  const int kYOffset = new_subdomain->getOffset(1);
  const int kYLength = new_subdomain->getLength(1);
  const int kXOffset = new_subdomain->getOffset(2);
  const int kXLength = new_subdomain->getLength(2);
  const int kUnknown = 1;

  // create new block
  new_subdomain = new SubDomain(id, kZOffset, num_elements_z, kYOffset,
          kYLength, kXOffset, kXLength, kUnknown, kUnknown, kUnknown,
          fake_neighbors);
  
  // copy blocks into new block memory
  int start_z_offset = domain.at(0)->getOffset(0);
  for (int i = 1; i < kNumBlocksToCombine; ++i)
    start_z_offset = std::min(start_z_offset, domain.at(i)->getOffset(0));

  for (unsigned int block_index = kFirstBlockIndex;
          block_index < kNumBlocksToCombine;
          ++block_index) {
    boost::scoped_ptr<SubDomain> block(domain.front());
    DTYPE* dest_begin = NULL;
    DTYPE* src_begin = NULL;
    DTYPE* src_end = NULL;
    int z_offset = 0;
  
    // Copy the leading ghost zone
    if (kFirstBlockIndex == block_index) {
      dest_begin = new_subdomain->getBuffer();
      src_begin = block->getBuffer();
      src_end = src_begin +
              kBorderZ * block->getLength(1) * block->getLength(2);
      std::copy(src_begin, src_end, dest_begin);
    } else if (kNumBlocksToCombine - 1 == block_index) {  // copy trailing gz
      z_offset = block->getOffset(0) - start_z_offset + block->getLength(0)
                  - kBorderZ;
      dest_begin = new_subdomain->getBuffer() + z_offset *
              new_subdomain->getLength(1) * new_subdomain->getLength(2);
      z_offset = block->getLength(0) - kBorderZ;
      src_begin = block->getBuffer() +
              z_offset * block->getLength(1) * block->getLength(2);
      src_end = src_begin + 
              kBorderZ * block->getLength(1) * block->getLength(2);
      std::copy(src_begin, src_end, dest_begin);
    }
    
    // Copy the middle data for every block
    z_offset = block->getOffset(0) - start_z_offset + kBorderZ;
    dest_begin = new_subdomain->getBuffer() +
            z_offset * new_subdomain->getLength(1) *
            new_subdomain->getLength(2);
    src_begin = block->getBuffer() + kBorderZ * block->getLength(1) *
            block->getLength(2);
    int z_length = block->getLength(0) - 2 * kBorderZ;

    src_end = src_begin + z_length * block->getLength(1) *
            block->getLength(2);
    std::copy(src_begin, src_end, dest_begin);
    
    domain.pop_front();
  }
  return new_subdomain;
}

/**
 * 
 * @param index index of sub_domain to return
 * @return pointer to specified sub_domain
 */
SubDomain* Decomposition::getSubDomain(const int index) {
  return this->domain.at(index);
}

size_t Decomposition::getNumSubDomains() const {
  return domain.size();
}

void Decomposition::addSubDomain(SubDomain* s) {
  domain.push_back(s);
}

void Decomposition::copyBlock3D(DTYPE* buffer, SubDomain* s,
        const int kDepth, const int kRows, const int kCols) {
  // subDomain should already have memory allocated.
  DTYPE *sBuff = s->getBuffer();
  if (NULL == sBuff) {
    fprintf(stderr, "copyBlock: subDomain has NULL Buffer.\n");
    return;
  }

  // stage sub-domain data into contiguous memory
  for (int depth = 0; depth < s->getLength(0); depth++) {
    for (int row = 0; row < s->getLength(1); row++) {
      for (int col = 0; col < s->getLength(2); col++) {
        // make sure index is valid
        int d = s->getOffset(0) + depth;
        int r = s->getOffset(1) + row;
        int c = s->getOffset(2) + col;

        // don't copy if we aren't inside valid range
        if (d < 0) d = 0;
        if (r < 0) r = 0;
        if (c < 0) c = 0;
        if (d >= kDepth) d = kDepth - 1;
        if (r >= kRows) r = kRows - 1;
        if (c >= kCols) c = kCols - 1;

        int newIndex = (depth * s->getLength(1) + row) * s->getLength(2) + col;
        int oldIndex = (d * kRows + r) * kCols + c;
        sBuff[newIndex] = buffer[oldIndex];
      }
    }
  }
} // end copyBlock3D

void Decomposition::copyBlock2D(DTYPE* buffer, SubDomain* s,
        const int numElementsRows, const int numElementsCols) {
  //subDomain should already have memory allocated.
  DTYPE *sBuff = s->getBuffer();
  if (NULL == sBuff) {
    fprintf(stderr, "copyBlock: subDomain has NULL Buffer.\n");
    return;
  }

  //stage sub-domain data into contiguous memory
  for (int row = 0; row < s->getLength(0); row++) {
    for (int col = 0; col < s->getLength(1); col++) {
      //make sure index is valid
      int r = s->getOffset(0) + row;
      int c = s->getOffset(1) + col;

      if (r < 0) r = 0;
      if (r >= numElementsRows) r = numElementsRows - 1;
      if (c < 0) c = 0;
      if (c >= numElementsCols) c = numElementsCols - 1;

      int newIndex = row * s->getLength(1) + col;
      int oldIndex = r * numElementsCols + c;
      sBuff[newIndex] = buffer[oldIndex];
    }
  }
}//end copyBlock2D

void Decomposition::decompose2D(DTYPE* buffer, const int numElementsRows,
        const int kNumElementsCols, const int stencil_size[3],
        const int kPyramidHeight, const int kNumberBlocksPerDimension) {

  int blockDimHeight = static_cast<int> ((numElementsRows /
          (double) kNumberBlocksPerDimension) + .5);
  int blockDimWidth = static_cast<int> ((kNumElementsCols /
          (double) kNumberBlocksPerDimension) + .5);

  //calculate ghost zone
  int border[2] = {kPyramidHeight * stencil_size[0],
    kPyramidHeight * stencil_size[1]};

  domain.clear();
  for (int i = 0; i < kNumberBlocksPerDimension; ++i) {
    for (int j = 0; j < kNumberBlocksPerDimension; ++j) {
      int id[2] = {i, j};

      //offset to account for ghost zone, may be negative
      int heightOff = blockDimHeight * i - border[0];
      int widthOff = blockDimWidth * j - border[1];

      //length may be too large when added to offset
      int heightLen = blockDimHeight + 2 * border[0];
      int widthLen = blockDimWidth + 2 * border[1];
      int fakeNeighbors[8] = {0};
      SubDomain* s = NULL;
      s = new SubDomain(id, heightOff, heightLen, widthOff, widthLen,
              kNumberBlocksPerDimension, kNumberBlocksPerDimension,
              fakeNeighbors);

      //get data for this block from the buffer
      copyBlock2D(buffer, s, numElementsRows, kNumElementsCols);
      domain.push_back(s);
      s = NULL;
    }
  }
}//end decompose2D

void Decomposition::decompose3DSlab(DTYPE* buffer, const int numElementsDepth,
        const int numElementsRows, const int numElementsCols,
        const int stencil_size[3], const int pyramidHeight,
        const int kNumberSlabs) {

  const int kSlabDepth = static_cast<int> ((numElementsDepth /
          (double) kNumberSlabs) + .5);

  //calculate ghost zone
  const int kBorder[3] = {pyramidHeight * stencil_size[0],
    pyramidHeight * stencil_size[1],
    pyramidHeight * stencil_size[2]};

  domain.clear();
  int j = 0, k = 0;
  for (int i = 0; i < kNumberSlabs; ++i) {
    int id[3] = {i, j, k};

    // offset to account for ghost zone, may be negative
    int depthOff = kSlabDepth * i - kBorder[0];
    int heightOff = -1 * kBorder[1];
    int widthOff = -1 * kBorder[2];

    // length may be too large when added to offset
    int depthLen = kSlabDepth + 2 * kBorder[0];
    int fakeNeighbors[26] = {0};
    SubDomain* block = NULL;
    block = new SubDomain(id, depthOff, depthLen, heightOff,
            numElementsRows, widthOff, numElementsCols,
            kNumberSlabs, 1, 1, fakeNeighbors);

    block->setBorder(0, kBorder[0]);
    block->setBorder(1, kBorder[1]);
    block->setBorder(2, kBorder[2]);

    // get data for this block from the buffer
    copyBlock3D(buffer, block, numElementsDepth, numElementsRows,
            numElementsCols);
    domain.push_back(block);
  }
} // end decompose3DSlab

void Decomposition::decompose3D(DTYPE* buffer, const int numElementsDepth,
        const int numElementsRows, const int numElementsCols,
        const int stencil_size[3], const int pyramidHeight,
        const int kNumberBlocksPerDimension) {

  int blockDimDepth = static_cast<int> ((numElementsDepth /
          (double) kNumberBlocksPerDimension) + .5);
  int blockDimHeight = static_cast<int> ((numElementsRows /
          (double) kNumberBlocksPerDimension) + .5);
  int blockDimWidth = static_cast<int> ((numElementsCols /
          (double) kNumberBlocksPerDimension) + .5);

  //calculate ghost zone
  int border[3] = {pyramidHeight * stencil_size[0],
    pyramidHeight * stencil_size[1],
    pyramidHeight * stencil_size[2]};

  domain.clear();
  for (int i = 0; i < kNumberBlocksPerDimension; ++i) {
    for (int j = 0; j < kNumberBlocksPerDimension; ++j) {
      for (int k = 0; k < kNumberBlocksPerDimension; ++k) {
        int id[3] = {i, j, k};

        //offset to account for ghost zone, may be negative
        int depthOff = blockDimDepth * i - border[0];
        int heightOff = blockDimHeight * j - border[1];
        int widthOff = blockDimWidth * k - border[2];

        //length may be too large when added to offset
        int depthLen = blockDimDepth + 2 * border[0];
        int heightLen = blockDimHeight + 2 * border[1];
        int widthLen = blockDimWidth + 2 * border[2];
        int fakeNeighbors[26] = {0};
        SubDomain* block = NULL;
        block = new SubDomain(id, depthOff, depthLen, heightOff, heightLen,
                widthOff, widthLen, kNumberBlocksPerDimension, kNumberBlocksPerDimension, kNumberBlocksPerDimension,
                fakeNeighbors);

        //get data for this block from the buffer
        copyBlock3D(buffer, block, numElementsDepth, numElementsRows,
                numElementsCols);
        domain.push_back(block);
      }
    }
  }
}//end decompose3D

void Decomposition::decompose(DTYPE* buffer, const int numDimensions,
        const int numElements[3], const int stencil_size[3],
        const int pyramidHeight, const int kNumberBlocksPerDimension) {
  if (2 == numDimensions)
    decompose2D(buffer, numElements[0], numElements[1], stencil_size,
          pyramidHeight, kNumberBlocksPerDimension);
  else if (3 == numDimensions)
    decompose3DSlab(buffer, numElements[0], numElements[1], numElements[2],
          stencil_size, pyramidHeight, kNumberBlocksPerDimension);
}

void printDecomposition(Decomposition& d) {
  for (size_t s = 0; s < d.getNumSubDomains(); ++s) {
    fprintf(stderr, "s[%zu] id[%d][%d][%d] lin_index[%d].\n",
            s, d.getSubDomain(s)->getId()[0], d.getSubDomain(s)->getId()[1],
            d.getSubDomain(s)->getId()[2], d.getSubDomain(s)->getLinIndex());
  }
}
