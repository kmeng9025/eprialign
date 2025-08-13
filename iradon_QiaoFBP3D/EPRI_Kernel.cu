__global__ void EPRI_3D_kernel(double *object,double *projection,double *GX, double *GY,double *GZ,double length_of_finalcube,double number_of_finalimage,double length_of_projection,double number_of_angle,double number_of_projection,double interp_method)
{


double xx,yy,zz,t,t0,value1,value2;
int kk,m,n,k,t1,t2;


 m=threadIdx.x;
 n=blockIdx.x;
 k=blockIdx.y;

 xx=(m+1-number_of_finalimage/2)*length_of_finalcube/number_of_finalimage;
 yy=(n+1-number_of_finalimage/2)*length_of_finalcube/number_of_finalimage;
 zz=(k+1-number_of_finalimage/2)*length_of_finalcube/number_of_finalimage;
                
                





for(kk=1;kk<=number_of_angle;kk++)
{                             
 if(int(interp_method)==0)
 {t=xx*GX[kk-1]+yy*GY[kk-1]+zz*GZ[kk-1];
 t=round(t/(length_of_projection/number_of_projection));
 t=t+(number_of_projection/2);
 if(t>=0&&t<=(number_of_projection-1))
 object[(int)(k*number_of_finalimage*number_of_finalimage+n*number_of_finalimage+m)]+=projection[(int)((kk-1)*number_of_projection+t)];
}
else 
{t=xx*GX[kk-1]+yy*GY[kk-1]+zz*GZ[kk-1];
 t0=t/(length_of_projection/number_of_projection);
 t1=floor(t/(length_of_projection/number_of_projection));
 t2=ceil(t/(length_of_projection/number_of_projection));
 t0=t0+(number_of_projection/2);
 t1=t1+(number_of_projection/2);
 t2=t2+(number_of_projection/2);

 if(t1>=0&&t2<=(number_of_projection-1))
     {value1=projection[(int)((kk-1)*number_of_projection+t1)];
      value2=projection[(int)((kk-1)*number_of_projection+t2)];
      object[(int)(k*number_of_finalimage*number_of_finalimage+n*number_of_finalimage+m)]+=value1+(value2-value1)*(t0-t1);
     }
}
}











}