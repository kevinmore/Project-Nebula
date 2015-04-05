#include "ShadowMapFBO.h"
#include <assert.h>
#include <QDebug>

ShadowMapFBO::ShadowMapFBO()
	: m_fbo(0),
	  m_shadowMap(0)
{
	Q_ASSERT(initializeOpenGLFunctions());
}

ShadowMapFBO::~ShadowMapFBO()
{
	destroy();
}

bool ShadowMapFBO::init( uint windowWidth, uint windowHeight )
{
	// Create the FBO
	glGenFramebuffers(1, &m_fbo);    

	// Create the depth buffer
	glGenTextures(1, &m_shadowMap);
	glBindTexture(GL_TEXTURE_2D, m_shadowMap);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, windowWidth, windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_shadowMap, 0);

	// Disable writes to the color buffer
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	GLenum Status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

	if (Status != GL_FRAMEBUFFER_COMPLETE) 
	{
		qWarning() << "Frame buffer error, status:" << Status; // 0x%x;
		return false;
	}

	return true;
}

void ShadowMapFBO::bindForWriting()
{
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fbo);
}

void ShadowMapFBO::bindForReading( GLenum textureUnit )
{
	glActiveTexture(textureUnit);
	glBindTexture(GL_TEXTURE_2D, m_shadowMap);
}

void ShadowMapFBO::release()
{
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

void ShadowMapFBO::destroy()
{
	if (m_shadowMap)
		glDeleteTextures(1, &m_shadowMap);

	if (m_fbo)
		glDeleteFramebuffers(1, &m_fbo);
}
