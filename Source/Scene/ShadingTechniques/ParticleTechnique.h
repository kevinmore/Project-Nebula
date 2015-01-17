#pragma once

#include "Technique.h"

#define NUM_PARTICLE_ATTRIBUTES 6
#define MAX_PARTICLES_ON_SCENE 100000

#define PARTICLE_TYPE_GENERATOR 0
#define PARTICLE_TYPE_NORMAL 1

class ParticleTechnique : public Technique 
{
public:

	enum ShaderType
	{
		UPDATE,
		RENDER
	};

	ParticleTechnique(const QString &shaderName, ShaderType shaderType);
	~ParticleTechnique() {}
	virtual bool init();

private:

	virtual bool compileShader();

	ShaderType m_shaderType;
};

