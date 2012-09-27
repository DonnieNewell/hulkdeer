#include "Decomposition.h"
#include <stdio.h>
Decomposition::Decomposition(){ }

Decomposition::Decomposition(const Decomposition &d){ *this = d;  }

Decomposition::~Decomposition() {
  for (size_t s = 0; s < domain.size(); ++s) {
    if (NULL != domain.at(s)) {
      delete domain.at(s);
      domain.at(s)=NULL;
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
    fprintf(stderr,"copyBlock: subDomain has NULL Buffer.\n");
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
}  // end copyBlock3D

void Decomposition::copyBlock2D(DTYPE* buffer, SubDomain* s,
        const int numElementsRows, const int numElementsCols) {
  //subDomain should already have memory allocated.
  DTYPE *sBuff = s->getBuffer();
  if (NULL == sBuff) {
    fprintf(stderr,"copyBlock: subDomain has NULL Buffer.\n");
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

        int newIndex =  row * s->getLength(1) + col;
        int oldIndex =  r * numElementsCols + c;
        sBuff[newIndex] = buffer[oldIndex];
    }
  }
}//end copyBlock2D

void Decomposition::decompose2D(DTYPE* buffer, const int numElementsRows,
        const int kNumElementsCols, const int stencil_size[3],
        const int kPyramidHeight, const int kNumberBlocksPerDimension) {

  int blockDimHeight  = static_cast<int>((numElementsRows /
                                    (double)kNumberBlocksPerDimension) + .5);
  int blockDimWidth   = static_cast<int>((kNumElementsCols /
                                    (double)kNumberBlocksPerDimension) + .5);

  //calculate ghost zone
  int border[2] = { kPyramidHeight * stencil_size[0],
                    kPyramidHeight * stencil_size[1]};

  domain.clear();
  for (int i = 0; i < kNumberBlocksPerDimension; ++i) {
    for (int j = 0; j < kNumberBlocksPerDimension; ++j) {
        int id[2] = {i, j};

        //offset to account for ghost zone, may be negative
        int heightOff = blockDimHeight * i - border[0];
        int widthOff  = blockDimWidth  * j - border[1];

        //length may be too large when added to offset
        int heightLen = blockDimHeight + 2 * border[0];
        int widthLen  = blockDimWidth + 2 * border[1];
        int fakeNeighbors[8] = {0};
        SubDomain* s = NULL;
        s= new SubDomain(id,heightOff, heightLen, widthOff, widthLen,
                kNumberBlocksPerDimension, kNumberBlocksPerDimension,
                fakeNeighbors);

        //get data for this block from the buffer
        copyBlock2D(buffer, s, numElementsRows, kNumElementsCols);
        domain.push_back(s);
        s=NULL;
      }
    }
}//end decompose2D

void Decomposition::decompose3DSlab(DTYPE* buffer, const int numElementsDepth,
        const int numElementsRows, const int numElementsCols,
        const int stencil_size[3], const int pyramidHeight,
        const int kNumberSlabs) {

  int slabDepth   = static_cast<int>((numElementsDepth /
                                    (double)kNumberSlabs) + .5);

  //calculate ghost zone
  int border[3] = { pyramidHeight * stencil_size[0],
                    pyramidHeight * stencil_size[1],
                    pyramidHeight * stencil_size[2]};

  domain.clear();
  int j = 0, k = 0;
  for (int i = 0; i < kNumberSlabs; ++i) {
        int id[3] = {i, j, k};

        // offset to account for ghost zone, may be negative
        int depthOff  = slabDepth * i - border[0];
        int heightOff = -1 * border[1];
        int widthOff  = -1 * border[2];

        // length may be too large when added to offset
        int depthLen  = slabDepth + 2 * border[0];
        int fakeNeighbors[26] = {0};
        SubDomain* block = NULL;
        block = new SubDomain(id, depthOff, depthLen, heightOff,
                            numElementsRows, widthOff, numElementsCols,
                            kNumberSlabs, 1, 1, fakeNeighbors);

        // get data for this block from the buffer
        copyBlock3D(buffer, block, numElementsDepth, numElementsRows,
                    numElementsCols);
        domain.push_back(block);
  }
}  // end decompose3D


void Decomposition::decompose3D(DTYPE* buffer, const int numElementsDepth,
        const int numElementsRows, const int numElementsCols,
        const int stencil_size[3], const int pyramidHeight,
        const int kNumberBlocksPerDimension) {

  int blockDimDepth   = static_cast<int>((numElementsDepth /
                                    (double)kNumberBlocksPerDimension) + .5);
  int blockDimHeight  = static_cast<int>((numElementsRows /
                                    (double)kNumberBlocksPerDimension) + .5);
  int blockDimWidth   = static_cast<int>((numElementsCols /
                                    (double)kNumberBlocksPerDimension) + .5);

  //calculate ghost zone
  int border[3] = { pyramidHeight * stencil_size[0],
                    pyramidHeight * stencil_size[1],
                    pyramidHeight * stencil_size[2]};

  domain.clear();
  for (int i = 0; i < kNumberBlocksPerDimension; ++i) {
    for (int j = 0; j < kNumberBlocksPerDimension; ++j) {
      for (int k = 0; k < kNumberBlocksPerDimension; ++k) {
        int id[3] = {i, j, k};

        //offset to account for ghost zone, may be negative
        int depthOff  = blockDimDepth  * i - border[0];
        int heightOff = blockDimHeight * j - border[1];
        int widthOff  = blockDimWidth  * k - border[2];

        //length may be too large when added to offset
        int depthLen  = blockDimDepth   + 2 * border[0];
        int heightLen = blockDimHeight  + 2 * border[1];
        int widthLen  = blockDimWidth   + 2 * border[2];
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
