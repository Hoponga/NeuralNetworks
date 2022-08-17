#include <windows.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <errno.h>
#include <fstream>
#include <string>

#define BitCountToColors(x) (((x) == 1) ? 2 : (((x) == 4) ? 16 : (((x) == 8) ? 256 : 0)))

using namespace std; 

int main(int argc, char* argv[])
   {
   char *pFileName;
   char *writeOutFileName = "out";
   FILE *pFile;
   BITMAPFILEHEADER bmpFileHeader;
   BITMAPINFOHEADER bmpInfoHeader;
   RGBQUAD rgbQuad;
   DWORD I;

   if (argc < 3)
      {
      if (argc ==1)
         pFileName = "test1.bmp";
      else
         pFileName = argv[1];
/*
typedef struct tagBITMAPFILEHEADER {
        WORD    bfType;
        DWORD   bfSize;
        WORD    bfReserved1;
        WORD    bfReserved2;
        DWORD   bfOffBits;
} BITMAPFILEHEADER, FAR *LPBITMAPFILEHEADER, *PBITMAPFILEHEADER;
*/

      if (pFile = fopen(pFileName, "rb"))
         {
         printf("BITMAPFILEHEADER = %d bytes\n", sizeof(BITMAPFILEHEADER));

         fread((char *)&bmpFileHeader, sizeof(BITMAPFILEHEADER), 1, pFile);

         printf("bfType=%2X bfSize=%lu bfReserved1=%u bfReserved2=%u bfOffBits=%lu\n",
                bmpFileHeader.bfType,
                bmpFileHeader.bfSize,
                bmpFileHeader.bfReserved1,
                bmpFileHeader.bfReserved2,
                bmpFileHeader.bfOffBits);
/*
typedef struct tagBITMAPINFOHEADER{
        DWORD      biSize;
        LONG       biWidth;
        LONG       biHeight;
        WORD       biPlanes;
        WORD       biBitCount;
        DWORD      biCompression;
        DWORD      biSizeImage;
        LONG       biXPelsPerMeter;
        LONG       biYPelsPerMeter;
        DWORD      biClrUsed;
        DWORD      biClrImportant;
} BITMAPINFOHEADER, FAR *LPBITMAPINFOHEADER, *PBITMAPINFOHEADER;
*/
         printf("BITMAPINFOHEADER = %d bytes\n", sizeof(BITMAPINFOHEADER));

         fread((char *)&bmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, pFile);
         //bmpInfoHeader.biWidth = 50;
         //bmpInfoHeader.biHeight = 50;

         printf("biSize=%lu\nbiWidth=%ld\nbiHeight=%ld\nbiPlanes=%u\nbiBitCount=%u\nbiCompression=%lu\nbiSizeImage=%lu\nbiXPelsPerMeter=%ld\nbiYPelsPerMeter=%ld\nbiClrUsed=%lu\nbiClrImportant=%lu\n",
                bmpInfoHeader.biSize,
                bmpInfoHeader.biWidth,
                bmpInfoHeader.biHeight,
                bmpInfoHeader.biPlanes,
                bmpInfoHeader.biBitCount,
                bmpInfoHeader.biCompression,
                bmpInfoHeader.biSizeImage,
                bmpInfoHeader.biXPelsPerMeter,
                bmpInfoHeader.biYPelsPerMeter,
                bmpInfoHeader.biClrUsed,
                bmpInfoHeader.biClrImportant);
         


         printf("\n");

/*
typedef struct tagRGBQUAD {
        BYTE    rgbBlue;
        BYTE    rgbGreen;
        BYTE    rgbRed;
        BYTE    rgbReserved;
} RGBQUAD;
typedef RGBQUAD FAR* LPRGBQUAD;
*/
         if (bmpInfoHeader.biBitCount <= 8)
            {
            printf("RGBQUAD size = %d\n", sizeof(RGBQUAD));

            for (I = 0; I < (DWORD)BitCountToColors(bmpInfoHeader.biBitCount); ++I)
               {
               fread((char *)&rgbQuad, sizeof(RGBQUAD), 1, pFile);
               printf("%d\t%d\t%d\t%d\n",rgbQuad.rgbBlue, rgbQuad.rgbGreen, rgbQuad.rgbRed, rgbQuad.rgbReserved);
               }

            }

         printf("\nPosition = %ld\n", ftell(pFile));
         string out = string(pFileName);
         out = out.substr(0, out.find('.'));
         cout << "File output: " << out << endl;

         std::ofstream fileOut(out);
         int sum; 

         for (I = 0; I < bmpInfoHeader.biWidth*bmpInfoHeader.biHeight; ++I)
            {
            fread((char *)&rgbQuad, sizeof(RGBQUAD), 1, pFile);
            sum = (0.3*(1.0*rgbQuad.rgbRed) + 0.59*(1.0*rgbQuad.rgbGreen)+0.11*(1.0*rgbQuad.rgbBlue));
            sum /= (255.0); 
            fileOut << sum << " ";
            }

         fclose(pFile);
         fileOut.close();

         }
      else
         perror(argv[1]);
      }
   else
      printf("syntax:\nbitmap file\n");

   return(0);
   }

/* Bitmap Header Definition
typedef struct tagBITMAP
  {
    LONG        bmType;
    LONG        bmWidth;
    LONG        bmHeight;
    LONG        bmWidthBytes;
    WORD        bmPlanes;
    WORD        bmBitsPixel;
    LPVOID      bmBits;
  } BITMAP, *PBITMAP, NEAR *NPBITMAP, FAR *LPBITMAP;
*/

/* structures for defining DIBs
typedef struct tagBITMAPCOREHEADER {
        DWORD   bcSize;                 // used to get to color table
        WORD    bcWidth;
        WORD    bcHeight;
        WORD    bcPlanes;
        WORD    bcBitCount;
} BITMAPCOREHEADER, FAR *LPBITMAPCOREHEADER, *PBITMAPCOREHEADER;
*/

/*
typedef struct tagBITMAPINFO {
    BITMAPINFOHEADER    bmiHeader;
    RGBQUAD             bmiColors[1];
} BITMAPINFO, FAR *LPBITMAPINFO, *PBITMAPINFO;

typedef struct tagBITMAPCOREINFO {
    BITMAPCOREHEADER    bmciHeader;
    RGBTRIPLE           bmciColors[1];
} BITMAPCOREINFO, FAR *LPBITMAPCOREINFO, *PBITMAPCOREINFO;
*/
