#include "ParticleSystem.h"
#include <Scene/Scene.h>
#include <Utility/Math.h>

ParticleSystem::ParticleSystem(Scene* scene)
	: Component(true, 1),// get rendered last
	  m_curReadBufferIndex(0),
	  m_aliveParticles(0),
	  fElapsedTime(0.0f),
	  m_scene(scene),
	  bInitialized(false)
{
}

ParticleSystem::~ParticleSystem()
{}

void ParticleSystem::initParticleSystem()
{
	Q_ASSERT(initializeOpenGLFunctions());	

	installShaders();

	prepareTransformFeedback();

	// load texture
	m_Texture = m_scene->textureManager()->addTexture("particle", "../Resource/Textures/particle.bmp");
}


void ParticleSystem::installShaders()
{
	// Updating program
	particleUpdater = ParticleTechniquePtr(new ParticleTechnique("particles_update", ParticleTechnique::UPDATE));
	if (!particleUpdater->init()) 
	{
		qWarning() << "particles_update initializing failed.";
		return;
	}

	// Rendering program
	particleRenderer = ParticleTechniquePtr(new ParticleTechnique("particles_render", ParticleTechnique::RENDER));
	if (!particleRenderer->init()) 
	{
		qWarning() << "particles_render initializing failed.";
		return;
	}

	bInitialized = true;
}

void ParticleSystem::prepareTransformFeedback()
{
	if(!bInitialized) return;

	glGenTransformFeedbacks(1, &m_transformFeedbackBuffer);
	glGenQueries(1, &uiQuery);

	glGenVertexArrays(2, m_VAO);
	glGenBuffers(2, m_particleBuffer);

	CParticle firstParticle;
	firstParticle.iType = PARTICLE_TYPE_GENERATOR;
	m_aliveParticles = 1;

	for (int i = 0; i < 2; ++i)
	{
		glBindVertexArray(m_VAO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, m_particleBuffer[i]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(CParticle)*MAX_PARTICLES_ON_SCENE, NULL, GL_DYNAMIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(CParticle), &firstParticle);

		for (int j = 0; j < NUM_PARTICLE_ATTRIBUTES; ++j)
		{
			glEnableVertexAttribArray(j);
		}


		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)0); // Position
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)12); // Velocity
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)24); // Color
		glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)36); // Lifetime
		glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)40); // Size
		glVertexAttribPointer(5, 1, GL_INT,	  GL_FALSE, sizeof(CParticle), (const GLvoid*)44); // Type
	}

	particleUpdater->setVAO(m_VAO[0]);
	particleRenderer->setVAO(m_VAO[1]);
}


void ParticleSystem::updateParticles( float fTimePassed )
{
	particleUpdater->enable();
	vGenPosition = m_actor->position();
	vec3 rot = m_actor->rotation();
	QQuaternion rotation = QQuaternion::fromAxisAndAngle(Math::Vector3D::UNIT_X, rot.x())
						 * QQuaternion::fromAxisAndAngle(Math::Vector3D::UNIT_Y, rot.y())
						 * QQuaternion::fromAxisAndAngle(Math::Vector3D::UNIT_Z, rot.z());

	particleUpdater->getShaderProgram()->setUniformValue("fTimePassed", fTimePassed);
	particleUpdater->getShaderProgram()->setUniformValue("fParticleMass", m_fParticleMass);
	particleUpdater->getShaderProgram()->setUniformValue("vGenPosition", vGenPosition);
	particleUpdater->getShaderProgram()->setUniformValue("vGenVelocityMin", rotation.rotatedVector(m_minVelocity));
	particleUpdater->getShaderProgram()->setUniformValue("vGenVelocityRange", rotation.rotatedVector(vGenVelocityRange));
	particleUpdater->getShaderProgram()->setUniformValue("vGenColor", vGenColor);
	particleUpdater->getShaderProgram()->setUniformValue("vForce", m_force);

	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeMin", m_fMinLife);
	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeRange", fGenLifeRange);

	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeMin", m_fMinLife);
	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeRange", fGenLifeRange);

	particleUpdater->getShaderProgram()->setUniformValue("fGenSize", m_fGenSize);
	particleUpdater->getShaderProgram()->setUniformValue("iNumToGenerate", 0);

	if (fElapsedTime > m_fEmitRate)
	{
		particleUpdater->getShaderProgram()->setUniformValue("iNumToGenerate", m_EmitAmount);
		fElapsedTime -= m_fEmitRate;

		vec3 vRandomSeed =  Math::Random::randUnitVec3();
		particleUpdater->getShaderProgram()->setUniformValue("vRandomSeed", vRandomSeed);
	}

	glEnable(GL_RASTERIZER_DISCARD);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_transformFeedbackBuffer);

	glBindVertexArray(m_VAO[m_curReadBufferIndex]);
	glEnableVertexAttribArray(1); // Re-enable velocity

	// store the results of transform feedback
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_particleBuffer[1-m_curReadBufferIndex]);

	glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, uiQuery);
	glBeginTransformFeedback(GL_POINTS);

	glDrawArrays(GL_POINTS, 0, m_aliveParticles);

	glEndTransformFeedback();

	glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
	glGetQueryObjectiv(uiQuery, GL_QUERY_RESULT, &m_aliveParticles);

	m_curReadBufferIndex = 1-m_curReadBufferIndex;

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
}

void ParticleSystem::render(const float currentTime)
{
	if(!bInitialized) return;
	float dt = currentTime - fElapsedTime;
	fElapsedTime = currentTime;

	updateParticles(dt);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glDepthMask(0);

	glDisable(GL_RASTERIZER_DISCARD);
	particleRenderer->enable();
	m_Texture->bind(COLOR_TEXTURE_UNIT);

	mat4 matProjection = m_scene->getCamera()->projectionMatrix();
	mat4 matView = m_scene->getCamera()->viewMatrix();
	vQuad1 = vec3::crossProduct(m_scene->getCamera()->viewVector(), m_scene->getCamera()->upVector()).normalized();
	vQuad2 = vec3::crossProduct(m_scene->getCamera()->viewVector(), vQuad1).normalized();

	particleRenderer->getShaderProgram()->setUniformValue("matrices.mProj", matProjection);
	particleRenderer->getShaderProgram()->setUniformValue("matrices.mView", matView);
	particleRenderer->getShaderProgram()->setUniformValue("vQuad1", vQuad1);
	particleRenderer->getShaderProgram()->setUniformValue("vQuad2", vQuad2);
	particleRenderer->getShaderProgram()->setUniformValue("gSampler", COLOR_TEXTURE_UNIT);

	glBindVertexArray(m_VAO[m_curReadBufferIndex]);
	glDisableVertexAttribArray(1); // Disable velocity, because we don't need it for rendering

	glDrawArrays(GL_POINTS, 0, m_aliveParticles);
	glDepthMask(1);	
	glDisable(GL_BLEND);
}


void ParticleSystem::setEmitterProperties( float particleMass, vec3 a_vGenVelocityMin, vec3 a_vGenVelocityMax, vec3 a_vGenGravityVector, vec3 a_vGenColor, float a_fGenLifeMin, float a_fGenLifeMax, float a_fGenSize, float emitRate, int a_iNumToGenerate )
{
	m_fParticleMass = particleMass;
	
	m_minVelocity = a_vGenVelocityMin;
	m_maxVelocity = a_vGenVelocityMax;
	vGenVelocityRange = m_maxVelocity - m_minVelocity;

	m_force = a_vGenGravityVector;
	vGenColor = a_vGenColor;
	m_fGenSize = a_fGenSize;

	m_fMinLife = a_fGenLifeMin;
	m_fMaxLife = a_fGenLifeMax;
	fGenLifeRange = m_fMaxLife - m_fMinLife;

	m_fEmitRate = emitRate;

	m_EmitAmount = a_iNumToGenerate;
}

int ParticleSystem::getAliveParticles()
{
	return m_aliveParticles;
}
