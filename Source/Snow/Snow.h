#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <Primitives/Component.h>
#include <QVector>
#include <QSharedPointer>
#include <Scene/ShadingTechniques/ParticleTechnique.h>
#include "SnowParticle.h"
#include "Grid.h"

class Scene;
class GameObject;
typedef QSharedPointer<GameObject> GameObjectPtr;

class Snow : public Component, protected QOpenGLFunctions_4_3_Core
{
public:
	Snow();
	~Snow();

	virtual QString className() { return "Snow"; }
	virtual void render(const float currentTime);

	void clear();
	inline int size() const { return m_particles.size(); }
	inline void resize( int n ) { m_particles.resize(n); }

	SnowParticle* data() { return m_particles.data(); }
	const QVector<SnowParticle>& getParticles() const { return m_particles; }
	QVector<SnowParticle>& particles() { return m_particles; }

	bool hasBuffers() const;
	void buildBuffers();
	void deleteBuffers();

	GLuint vbo() { if ( !hasBuffers() ) buildBuffers(); return m_glVBO; }

	void merge( const Snow &particles ) { m_particles += particles.m_particles; deleteBuffers(); }

	Snow& operator += ( const Snow &particles ) { m_particles += particles.m_particles; deleteBuffers(); return *this; }
	Snow& operator += ( const SnowParticle &particle ) { m_particles.append(particle); deleteBuffers(); return *this; }

private:
	void installShader();
	QVector<SnowParticle> m_particles;
	Scene* m_scene;
	ParticleTechniquePtr m_snowEffect;
	GLuint m_glVBO;
	GLuint m_glVAO;
};

typedef QSharedPointer<Snow> SnowPtr;