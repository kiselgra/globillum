#pragma once
#include <iostream>
#include <fstream>
#include "materialBase.h"

namespace rta{
struct PrincipledBRDFParameters{
        float metallic ;
        float subsurface;
        float specular ;
        float roughness;
        float specularTint;
        float anisotropic;
        float sheen;
        float sheenTint;
        float clearcoat;
        float clearcoatGloss;
        float opacity;
        float transmiss;
        float3 color;
        float3 tangent;

        PrincipledBRDFParameters():metallic(0.0f),subsurface(0.0f),specular(0.0f),roughness(0.0f),specularTint(0.0f),
                anisotropic(0.0f),sheen(0.0f),sheenTint(0.0f),clearcoat(0.0f),clearcoatGloss(0.0f),opacity(1.0f),transmiss(1.0f){}

        PrincipledBRDFParameters(float3 &c):metallic(0.0f),subsurface(0.0f),specular(0.0f),roughness(0.0f),specularTint(0.0f),
                anisotropic(0.0f),sheen(0.0f),sheenTint(0.0f),clearcoat(0.0f),clearcoatGloss(0.0f),opacity(1.0f),color(c),transmiss(1.0f){}

        PrincipledBRDFParameters(const PrincipledBRDFParameters& other){
                metallic = other.metallic;
                subsurface = other.subsurface;
                specular = other.specular;
                roughness = other.roughness;
                specularTint = other.specularTint;
                anisotropic = other.anisotropic;
                sheen = other.sheen;
                sheenTint = other.sheenTint;
                clearcoat = other.clearcoat;
                clearcoatGloss = other.clearcoatGloss;
                color = other.color;
                tangent = other.tangent;
                opacity = other.opacity;
                transmiss = other.transmiss;
}

	PrincipledBRDFParameters(const std::string s){
	 	readFrom(s);
	}
inline void print(){
                std::cerr<<"METALLIC:"<<metallic<<"\n";
                std::cerr<<"subsurface:"<<subsurface<<"\n";
                std::cerr<<"specular:"<<specular<<"\n";
                std::cerr<<"roughness:"<<roughness<<"\n";
                std::cerr<<"specularTint:"<<specularTint<<"\n";
                std::cerr<<"anisotropic:"<<anisotropic<<"\n";
                std::cerr<<"sheen:"<<sheen<<"\n";
                std::cerr<<"sheenTint:"<<sheenTint<<"\n";
                std::cerr<<"clearcoat:"<<clearcoat<<"\n";
                std::cerr<<"clearcoatGloss:"<<clearcoatGloss<<"\n";
                std::cerr<<"color:"<<color.x <<","<< color.y <<","<< color.z <<"\n";
                std::cerr<<"opacity:"<<opacity<<"\n";
        }

        inline void readFrom(const std::string& filename){
                std::ifstream in(filename.c_str());
                while(!in.eof()){
                        std::string identifier;
                        in >> identifier;
                        if(!strcmp(identifier.c_str(),"color")){
                                in >> color.x;
                                in >> color.y;
                                in >> color.z;
                                continue;
                        }
                        if(!strcmp(identifier.c_str(),"metallic")){
                                in >> metallic;
                                continue;
                        }
                        if(!strcmp(identifier.c_str(),"opacity")){
                                in >> opacity;
                                continue;
                        }

			if(!strcmp(identifier.c_str(),"subsurface")){
                                in >> subsurface;
                                continue;
                        }
                        if(!strcmp(identifier.c_str(),"specular")){
                                in >> specular;
                                continue;
                        }
                        if(!strcmp(identifier.c_str(),"roughness")){
                                in >> roughness;
                                continue;
                        }
                        if(!strcmp(identifier.c_str(),"tintSpecular")){
                                in >> specularTint;
                                continue;
                        }
                        if(!strcmp(identifier.c_str(),"anisotropic")){
                                in >> anisotropic;
                                continue;
                        }
                        if(!strcmp(identifier.c_str(),"sheen")){
                                in >> sheen;
                                continue;
                        }
                        if(!strcmp(identifier.c_str(),"tintSheen")){
                                in >> sheenTint;
                                continue;
                        }
                        if(!strcmp(identifier.c_str(),"clearcoat")){
                                in >> clearcoat;
                                continue;
                        }
                        if(!strcmp(identifier.c_str(),"glossClearcoat")){
                                in >> clearcoatGloss;
                                continue;
                        }
                        if(!strcmp(identifier.c_str(),"transmission")){
                                in >> transmiss;
                                continue;
                        }
                }
        }
inline PrincipledBRDFParameters& operator *=(const float b ) {
                this->metallic *= b;
                this->subsurface *= b;
                this->specular *= b;
                this->roughness *= b;
                this->specularTint *= b;
                this->anisotropic *= b;
                this->sheen *= b;
                this->sheenTint *= b;
                this->clearcoat *= b;
                this->clearcoatGloss *= b;
                this->opacity *= b;
        //        this->tangent *= b;
         //       this->color *= b;
                this->transmiss *= b;
                return (*this);
        }

        inline PrincipledBRDFParameters& operator +=(const PrincipledBRDFParameters& b ) {
                this->metallic          += b.metallic           ;
                this->subsurface        += b.subsurface ;
                this->specular          += b.specular           ;
                this->roughness         += b.roughness          ;
                this->specularTint      += b.specularTint       ;
                this->anisotropic       += b.anisotropic        ;
                this->sheen                     += b.sheen                      ;
                this->sheenTint         += b.sheenTint          ;
                this->clearcoat         += b.clearcoat          ;
                this->clearcoatGloss+= b.clearcoatGloss;
                this->opacity += b.opacity;
           //     this->tangent           += b.tangent;
            //    this->color                     += b.color;
                this->transmiss += b.transmiss;
                return (*this);
        }
};

}
