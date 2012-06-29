#include "Decomposition.h"
#include <stdio.h>
Decomposition::Decomposition(){ }

Decomposition::Decomposition(const int numSubDomains){ }

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
        const int numElementsDepth, const int numElementsRows,
        const int numElementsCols) {
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
        if(d < 0 || d >= numElementsDepth || r < 0 || r >= numElementsRows ||
           c < 0 || c >= numElementsCols)
          continue;

        int newIndex =  depth * s->getLength(1) * s->getLength(2) +
                        row * s->getLength(0) + col ;
        int oldIndex =  d * numElementsCols * numElementsRows +
                        r * numElementsCols + c ;
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

        //don't copy if we aren't inside valid range
        if ( r < 0 || r >= numElementsRows || c < 0 || c >= numElementsCols)
          continue;

        int newIndex =  row * s->getLength(1) + col;
        int oldIndex =  r * numElementsCols + c;
        sBuff[newIndex] = buffer[oldIndex];
    }
  }
}//end copyBlock2D

void Decomposition::decompose1D(DTYPE* buffer, const int numElementsX,
        const int stencil_size[3], const int iterations) {
  if (0 < numElementsX) {
    int numLeftX=numElementsX;
    const int num_chunks = 64;
    int stride = numElementsX/num_chunks;
    for (size_t i=0; i < domain.size(); ++i) {
      SubDomain* s = domain.at(i);
      s->setLength(0, stride);
      s->setOffset(0, numElementsX - numLeftX);
      numLeftX -= s->getLength(0);
    } //end for
  }//end if
}//end decompose1D

void Decomposition::decompose2D(DTYPE* buffer, const int numElementsRows,
        const int numElementsCols, const int stencil_size[3],
        const int pyramidHeight) {
  int number_of_chunks = 8;
  int blockDimHeight  = static_cast<int>((numElementsRows /(double)number_of_chunks)+.5);
  int blockDimWidth   = static_cast<int>((numElementsCols /(double)number_of_chunks)+.5);

  //calculate ghost zone
  int border[2] = { pyramidHeight * stencil_size[0],
                    pyramidHeight * stencil_size[1]};

  domain.clear();
  for (int i = 0; i < number_of_chunks; ++i) {
    for (int j = 0; j < number_of_chunks; ++j) {
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
                number_of_chunks, number_of_chunks,
                fakeNeighbors);

        //get data for this block from the buffer
        copyBlock2D(buffer, s, numElementsRows, numElementsCols);
        domain.push_back(s);
        s=NULL;
      }
    }
}//end decompose2D

void Decomposition::decompose3D(DTYPE* buffer, const int numElementsDepth,
        const int numElementsRows, const int numElementsCols,
        const int stencil_size[3], const int pyramidHeight) {
  int number_of_chunks = 8;
  int blockDimDepth   = static_cast<int>((numElementsDepth/(double)number_of_chunks)+.5);
  int blockDimHeight  = static_cast<int>((numElementsRows /(double)number_of_chunks)+.5);
  int blockDimWidth   = static_cast<int>((numElementsCols /(double)number_of_chunks)+.5);

  //calculate ghost zone
  int border[3] = { pyramidHeight*stencil_size[0],
                    pyramidHeight*stencil_size[1],
                    pyramidHeight*stencil_size[2]};

  domain.clear();
  for (int i = 0; i < number_of_chunks; ++i) {
    for (int j = 0; j < number_of_chunks; ++j) {
      for (int k = 0; k < number_of_chunks; ++k) {
        int id[3] = {i, j, k};

        //offset to account for ghost zone, may be negative
        int depthOff  = blockDimDepth  * i - border[0];
        int heightOff = blockDimHeight * j - border[1];
        int widthOff  = blockDimWidth  * k - border[2];

        //length may be too large when added to offset
        int depthLen  = blockDimDepth   + 2*border[0];
        int heightLen = blockDimHeight  + 2*border[1];
        int widthLen  = blockDimWidth   + 2*border[2];
        int fakeNeighbors[26] = {0};
        SubDomain* s = NULL;
        s= new SubDomain(id, depthOff, depthLen, heightOff, heightLen,
                            widthOff, widthLen, number_of_chunks, number_of_chunks, number_of_chunks,
                            fakeNeighbors);

        //get data for this block from the buffer
        copyBlock3D(buffer, s, numElementsDepth, numElementsRows,
                  numElementsCols);
        domain.push_back(s);
        s=NULL;
      }
    }
  }
}//end decompose3D

void Decomposition::decompose(DTYPE* buffer, const int numDimensions,
        const int numElements[3], const int stencil_size[3],
        const int pyramidHeight) {
  if (1 == numDimensions)
    decompose1D(buffer, numElements[0], stencil_size, pyramidHeight);
  else if (2 == numDimensions)
    decompose2D(buffer, numElements[0], numElements[1], stencil_size, pyramidHeight);
  else if (3 == numDimensions)
    decompose3D(buffer, numElements[0], numElements[1], numElements[2],
                stencil_size, pyramidHeight);
}

void printDecomposition(Decomposition& d) {
  for (size_t s = 0; s < d.getNumSubDomains(); ++s) {
    fprintf(stderr, "s[%zu] off[%d][%d][%d] len[%d][%d][%d].\n",
            s, d.getSubDomain(s)->getOffset(0), d.getSubDomain(s)->getOffset(1),
            d.getSubDomain(s)->getOffset(2), d.getSubDomain(s)->getLength(0),
            d.getSubDomain(s)->getLength(1), d.getSubDomain(s)->getLength(2));
  }
}