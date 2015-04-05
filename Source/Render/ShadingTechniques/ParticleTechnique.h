#pragma once

#include "Technique.h"

#define NUM_PARTICLE_ATTRIBUTES 6
#define MAX_PARTICLES_PER_SYSTEM 100000

#define PARTICLE_TYPE_GENERATOR 0
#define PARTICLE_TYPE_NORMAL 1

class ParticleTechnique : public Technique 
{
public:

	enum ShaderType
	{
		UPDATE,
		RENDER,
		GENERAL
	};

	ParticleTechnique(const QString &shaderName, ShaderType shaderType);
	virtual bool init();

private:

	virtual bool compileShader();

	ShaderType m_shaderType;
};

typedef QSharedPointer<ParticleTechnique> ParticleTechniquePtr;