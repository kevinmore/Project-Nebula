/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Fwd/hkcstring.h>
#include <Common/Base/System/Error/hkFilterError.h>


void hkFilterError::install(Message m, int id, jumpbuf* jumpBuffer)
{
	if (m_prevHandler != HK_NULL) return;	
	m_prevHandler = &hkError::getInstance();
	m_prevHandler->addReference();
	hkError::replaceInstanceAndAddReference(this);

	m_message = m;
	m_id = id;
	m_jumpBuffer = jumpBuffer;
	m_messageRaised = false;
}

void hkFilterError::uninstall()
{
	if (m_prevHandler != HK_NULL)
	{
		//this->addReference();
		hkError::replaceInstanceAndAddReference(m_prevHandler);
		m_prevHandler->removeReference();
	}
}

int hkFilterError::message(Message m, int id, const char* description, const char* file, int line)
{
	if ((m == m_message) && (id == m_id)) 
	{
		m_messageRaised = true;

#if !defined(HK_PLATFORM_CTR)
		if ( m_jumpBuffer )
		{
			HK_STD_NAMESPACE::longjmp(*m_jumpBuffer, 1);
		}
#endif

		if (m_fwdFilteredMessage)
		{
			return m_prevHandler->message(m, id, description, file, line);
		}
		else
		{
			return 0;
		}
	}

	if (m_fwdOtherMessages)
	{
		return m_prevHandler->message(m, id, description, file, line);
	}
	else
	{
		return 0;
	}
}

void hkFilterError::setEnabled(int id, hkBool enabled)
{
	m_prevHandler->setEnabled(id, enabled);
}

hkBool hkFilterError::isEnabled(int id)
{
	return m_prevHandler->isEnabled(id);
}

void hkFilterError::enableAll()
{
	m_prevHandler->enableAll();
}

void hkFilterError::sectionBegin(int id, const char* sectionName)
{
	m_prevHandler->sectionBegin(id, sectionName);
}

void hkFilterError::sectionEnd()
{
	m_prevHandler->sectionEnd();
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
