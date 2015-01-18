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
	Q_OBJECT

public:
	ParticleSystem(Scene* scene);
	~ParticleSystem();

	void initParticleSystem();
	void installShaders();
	void prepareTransformFeedback();

	void updateParticles(float fTimePassed);
	virtual QString className() { return "ParticleSystem"; }
	virtual void render(const float currentTime);

	void setEmitterProperties(float particleMass, vec3 a_vGenVelocityMin, vec3 a_vGenVelocityMax, vec3 a_vGenGravityVector, vec3 a_vGenColor, float a_fGenLifeMin, float a_fGenLifeMax, float a_fGenSize, float fEvery, int a_iNumToGenerate);

	void ClearAllParticles();
	bool ReleaseParticleSystem();

	int getAliveParticles();

public slots:
	void setParticleMass(double m) { m_fParticleMass = (float)m; }
	void setParticleSize(double s) { m_fGenSize = (float)s; }
	void setEmitRate(double r)     { m_fEmitRate = (float)r; }
	void setEmitAmount(int a)	   { m_EmitAmount= a; }

	void setMinLife(double l) { m_fMinLife = l; fGenLifeRange = m_fMaxLife - m_fMinLife; }
	void setMaxLife(double l) { m_fMaxLife = l; fGenLifeRange = m_fMaxLife - m_fMinLife; }

	void setForceX(double f) { m_force.setX((float)f); }
	void setForceY(double f) { m_force.setY((float)f); }
	void setForceZ(double f) { m_force.setZ((float)f); }

	void setMinVelX(double v) { m_minVelocity.setX((float)v); vGenVelocityRange = m_maxVelocity - m_minVelocity;}
	void setMinVelY(double v) { m_minVelocity.setY((float)v); vGenVelocityRange = m_maxVelocity - m_minVelocity;}
	void setMinVelZ(double v) { m_minVelocity.setZ((float)v); vGenVelocityRange = m_maxVelocity - m_minVelocity;}

	void setMaxVelX(double v) { m_maxVelocity.setX((float)v); vGenVelocityRange = m_maxVelocity - m_minVelocity;}
	void setMaxVelY(double v) { m_maxVelocity.setY((float)v); vGenVelocityRange = m_maxVelocity - m_minVelocity;}
	void setMaxVelZ(double v) { m_maxVelocity.setZ((float)v); vGenVelocityRange = m_maxVelocity - m_minVelocity;}

public:
	float getParticleMass() const { return m_fParticleMass; }
	float getParticleSize() const { return m_fGenSize; }
	float getEmitRate()     const { return m_fEmitRate; }
	int   getEmitAmount()   const { return m_EmitAmount; }
	float getMinLife() const { return m_fMinLife; }
	float getMaxLife() const { return m_fMaxLife; }

	vec3 getForce()  const { return m_force; }
	vec3 getMinVel() const { return m_minVelocity; }
	vec3 getMaxVel() const { return m_maxVelocity; }

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
	float m_fEmitRate;

	vec3 vGenPosition;
	vec3 m_minVelocity, m_maxVelocity, vGenVelocityRange;
	vec3 m_force;
	vec3 vGenColor;

	float m_fMinLife, m_fMaxLife, fGenLifeRange;
	float m_fGenSize;
	float m_fParticleMass;

	int m_EmitAmount;

	ParticleTechniquePtr particleRenderer, particleUpdater;
	Scene* m_scene;
	TexturePtr m_Texture;
};

typedef QSharedPointer<ParticleSystem> ParticleSystemPtr;