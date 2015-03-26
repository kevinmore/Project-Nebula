#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <QColor>
#include <Utility/EngineCommon.h>
#include <Primitives/Component.h>
#include <Primitives/Texture.h>
#include <Scene/ShadingTechniques/ParticleTechnique.h>
class Particle
{
public:
	vec3 vPosition;
	vec3 vVelocity;
	vec3 vColor;
	float fLifeTime;
	float fSize;
	int iType;
};

class Scene;
class GameObject;
typedef QSharedPointer<GameObject> GameObjectPtr;
class ParticleSystem : public Component, protected QOpenGLFunctions_4_3_Core
{
	Q_OBJECT

public:
	ParticleSystem();
	~ParticleSystem();

	void initParticleSystem();
	void installShaders();
	void prepareTransformFeedback();

	void updateParticles(float fTimePassed);
	virtual QString className() { return "ParticleSystem"; }
	virtual void render(const float currentTime);

	void resetEmitter();
	void assingCollisionObject(GameObjectPtr collider);

	void ClearAllParticles();
	bool ReleaseParticleSystem();

	inline int getAliveParticlesCount() { return m_aliveParticlesCount; }
	inline vec3 getLinearImpuse() { return m_vLinearImpulse; }

public slots:
	void setParticleMass(double m) { m_particleMass = (float)m; }
	void setGravityFactor(double f){ m_gravityFactor = (float)f; }
	void setParticleSize(double s) { m_size = (float)s; }
	void setEmitRate(double r)     { m_emitRate = (float)r; }
	void setEmitAmount(int a)	   { m_emitAmount= a; }

	void setMinLife(double l) { m_minLife = l; m_lifeRange = m_maxLife - m_minLife; }
	void setMaxLife(double l) { m_maxLife = l; m_lifeRange = m_maxLife - m_minLife; }

	void setForceX(double f) { m_force.setX((float)f); }
	void setForceY(double f) { m_force.setY((float)f); }
	void setForceZ(double f) { m_force.setZ((float)f); }
	void setRestitution(double k) { m_restitution = (float)k; }

	void setMinVelX(double v) { m_minVelocity.setX((float)v); m_velocityRange = m_maxVelocity - m_minVelocity;}
	void setMinVelY(double v) { m_minVelocity.setY((float)v); m_velocityRange = m_maxVelocity - m_minVelocity;}
	void setMinVelZ(double v) { m_minVelocity.setZ((float)v); m_velocityRange = m_maxVelocity - m_minVelocity;}

	void setMaxVelX(double v) { m_maxVelocity.setX((float)v); m_velocityRange = m_maxVelocity - m_minVelocity;}
	void setMaxVelY(double v) { m_maxVelocity.setY((float)v); m_velocityRange = m_maxVelocity - m_minVelocity;}
	void setMaxVelZ(double v) { m_maxVelocity.setZ((float)v); m_velocityRange = m_maxVelocity - m_minVelocity;}

	void toggleRandomColor(bool state) { m_isColorRandomed = state; }
	void toggleCollision(bool state) { m_isCollisionEnabled = state; }
	void toggleImpulseGenerator(bool state) { m_generateImpulse = state; }

public:
	float getParticleMass() const { return m_particleMass; }
	float getGravityFactor()const { return m_gravityFactor; }
	float getParticleSize() const { return m_size; }
	float getEmitRate()     const { return m_emitRate; }
	int   getEmitAmount()   const { return m_emitAmount; }
	float getMinLife() const { return m_minLife; }
	float getMaxLife() const { return m_maxLife; }

	vec3 getForce()  const { return m_force; }
	void setForce(const vec3& f)  { m_force = f; }
	bool isCollisionEnabled() const { return m_isCollisionEnabled; }
	float getRestitution() const { return m_restitution; }

	vec3 getMinVel() const { return m_minVelocity; }
	void setMinVel(const vec3& v)  { m_minVelocity = v; m_velocityRange = m_maxVelocity - m_minVelocity; }

	vec3 getMaxVel() const { return m_maxVelocity; }
	void setMaxVel(const vec3& v)  { m_maxVelocity = v; m_velocityRange = m_maxVelocity - m_minVelocity; }

	bool isColorRandom() const { return m_isColorRandomed; }
	QColor getParticleColor() const;
	void setParticleColor(const QColor& col);

	QString getTextureFileName();
	TexturePtr getTexture() const { return m_Texture; }
	void loadTexture(const QString& fileName);

private:
	bool m_initialized;

	uint m_transformFeedbackBuffer;

	uint m_particleBuffer[2];
	uint m_VAO[2];

	uint m_query;
	uint m_textureID;

	int m_curReadBufferIndex;
	int m_aliveParticlesCount;

	vec3 vQuad1, vQuad2;

	float m_elapsedTime;
	float m_emitRate;
	float m_nextEmitTime;

	vec3 m_position;
	vec3 m_minVelocity, m_maxVelocity, m_velocityRange;
	vec3 m_force;

	GameObjectPtr m_collider;
	vec3 m_planePoint;
	vec3 m_planeNormal;
	bool m_isCollisionEnabled;
	float m_restitution;

	vec3 m_color;
	bool m_isColorRandomed;

	float m_minLife, m_maxLife, m_lifeRange;
	float m_size;
	float m_particleMass;
	float m_gravityFactor;

	int m_emitAmount;

	bool m_generateImpulse;
	vec3 m_vLinearImpulse;


	ParticleTechniquePtr particleRenderer, particleUpdater;
	Scene* m_scene;
	TexturePtr m_Texture;
};

typedef QSharedPointer<ParticleSystem> ParticleSystemPtr;