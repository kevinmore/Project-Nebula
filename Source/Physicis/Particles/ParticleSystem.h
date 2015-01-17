#pragma once
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_3_Core>
#include <Utility/EngineCommon.h>
#include <Primitives/Component.h>
#include <Primitives/Texture.h>
#include <Scene/ShadingTechniques/ParticleTechnique.h>

/*****************************************************

Class:		CParticle

Purpose:	Encapsulates particle and its properties.

*****************************************************/

class CParticle
{
public:
	vec3 vPosition;
	vec3 vVelocity;
	vec3 vColor;
	float fLifeTime;
	float fSize;
	int iType;
};

/**********************************************************************

Class:		CParticleSystemTransformFeedback

Purpose:	Particle system class that uses transform feedback feature.

***********************************************************************/

class Scene;
class ParticleSystem : public Component, protected QOpenGLFunctions_4_3_Core
{
public:
	ParticleSystem(Scene* scene);
	~ParticleSystem();

	void initParticleSystem();
	void installShaders();
	void prepareTransformFeedback();

	void updateParticles(float fTimePassed);
	virtual void render(const float currentTime);

	void setEmitterProperties(float particleMass, vec3 a_vGenVelocityMin, vec3 a_vGenVelocityMax, vec3 a_vGenGravityVector, vec3 a_vGenColor, float a_fGenLifeMin, float a_fGenLifeMax, float a_fGenSize, float fEvery, int a_iNumToGenerate);

	void ClearAllParticles();
	bool ReleaseParticleSystem();

	int getAliveParticles();

	

private:
	bool bInitialized;

	uint m_transformFeedbackBuffer;

	uint m_particleBuffer[2];
	uint m_VAO[2];

	uint uiQuery;
	uint uiTexture;

	int m_curReadBufferIndex;
	int m_aliveParticles;

	vec3 vQuad1, vQuad2;

	float fElapsedTime;
	float fNextGenerationTime;

	vec3 vGenPosition;
	vec3 vGenVelocityMin, vGenVelocityRange;
	vec3 vForce;
	vec3 vGenColor;

	float fGenLifeMin, fGenLifeRange;
	float fGenSize;
	float fParticleMass;

	int iNumToGenerate;

	ParticleTechniquePtr particleRenderer, particleUpdater;
	Scene* m_scene;
	TexturePtr m_Texture;
};

typedef QSharedPointer<ParticleSystem> ParticleSystemPtr;