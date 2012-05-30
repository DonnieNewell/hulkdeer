#ifndef SUB_DOMAIN_3D_H
#define SUB_DOMAIN_3D_H

#define DTYPE int
class SubDomain3D{
  int id[3];
  int offset[3];
  int length[3];
  int neighbors[26];//ranks of all neighbors to exchange ghost zones with
  //needs to be set by compiler. DTYPE maybe?
  DTYPE* buffer;
  public: 
    SubDomain3D();
    SubDomain3D(const SubDomain3D&);
    SubDomain3D(int[3],int, int, int, int, int, int);
    SubDomain3D& operator=(const SubDomain3D &);
    ~SubDomain3D();
    //needs to be set by compiler. DTYPE maybe?
    void setId(int,int,int);	
    void setLength(int, int);	
    void setOffset(int, int);
    //needs to be set by compiler. DTYPE maybe?
    DTYPE* getBuffer()const;	
    const int* getId()const;	
    int getLength(int)const;	
    int getOffset(int)const;
};

#endif
