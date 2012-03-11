#ifndef SUB_DOMAIN_3D_H
#define SUB_DOMAIN_3D_H

class SubDomain3D{
  int offset[3];
  int length[3];

  public: 
    SubDomain3D();
    SubDomain3D(const SubDomain3D&);
    SubDomain3D(int, int, int, int, int, int);
    SubDomain3D& operator=(const SubDomain3D &);
    ~SubDomain3D();
    void setLength(int, int);	
    void setOffset(int, int);
    int getLength(int)const;	
    int getOffset(int)const;
};

#endif
