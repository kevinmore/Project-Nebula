/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/System/Error/hkDefaultError.h>
#include <Common/Base/System/StackTracer/hkStackTracer.h>

hkDefaultError::hkDefaultError( hkErrorReportFunction errorReportFunction, void* errorReportObject )
:	m_errorFunction(errorReportFunction), 
	m_errorObject(errorReportObject), 
	m_minimumMessageLevel(hkError::MESSAGE_ALL)
{
#if defined(HK_PLATFORM_IOS)
	setEnabled(0x46AEFCEE, false);
	setEnabled(0xF021D445, false);
	setEnabled(0x4314aad9, false);
	setEnabled(0x2db3d51a, false);
#elif defined(HK_PLATFORM_CTR) || defined(HK_PLATFORM_ANDROID)
	setEnabled(0x64211c2f, false);
#endif
}

void hkDefaultError::setEnabled( int id, hkBool enabled )
{
	if( enabled )
	{
		m_disabledAssertIds.remove(id);
	}
	else
	{
		m_disabledAssertIds.insert(id, 1);
	}
}

hkBool hkDefaultError::isEnabled( int id )
{
	return m_disabledAssertIds.getWithDefault(id, 0) == 0;
}

void hkDefaultError::enableAll()
{
	m_minimumMessageLevel = hkError::MESSAGE_ALL;
	m_disabledAssertIds.clear();
}


void hkDefaultError::setMinimumMessageLevel( Message msg )
{
	m_minimumMessageLevel = msg;
}


hkError::Message hkDefaultError::getMinimumMessageLevel()
{
	return m_minimumMessageLevel;
}


void hkDefaultError::showMessage(const char* what, int id, const char* desc, const char* file, int line, hkBool stackTrace)
{
	const int buffer_size = 4*2048;
	char buffer[buffer_size];

	if ( ( id ==0 ) && ( file == HK_NULL ) )
	{
		//Just a simple message
		hkString::snprintf(buffer, buffer_size, "%s", desc);
	}
	else
	{
#if defined(HK_PLATFORM_LINUX)
		static const char format []     =  "%s:%d: [0x%08X] %s: %s";
		static const char formatNoId [] =  "%s:%d: %s: %s";
#else
		static const char format []     = "%s(%d): [0x%08X] %s: %s";	
		static const char formatNoId [] = "%s(%d): %s: %s";	
#endif

		if ( id != (int)0xffffffff && id!=0)
		{
			hkString::snprintf(buffer, buffer_size, format, file, line, id, what, desc);
		}
		else
		{
			hkString::snprintf(buffer, buffer_size, formatNoId, file, line, what, desc);
		}
	}

	// make sure the string is always terminated properly
	buffer[buffer_size-1] = 0;

	(*m_errorFunction)( buffer, m_errorObject );

	if( stackTrace )
	{
		hkStackTracer tracer;

		hkUlong trace[128];
		int ntrace = tracer.getStackTrace(trace, HK_COUNT_OF(trace) );
		if( ntrace > 2 )
		{
			(*m_errorFunction)("Stack trace is:\n", m_errorObject);
			// first two frames are in this file.
			tracer.dumpStackTrace(trace+2, ntrace-2, m_errorFunction, m_errorObject);
		}
	}
}



int hkDefaultError::message(hkError::Message msg, int id, const char* description, const char* file, int line)
{
	if( id == -1 && m_sectionIds.getSize() )
	{
		id = m_sectionIds.back();
	}

	if ( msg < getMinimumMessageLevel() )
	{
		return 0;
	}

	if (!isEnabled(id))
	{
		return 0;
	}

	const char* what = "";

	hkBool stackTrace = false;
	switch( msg )
	{
		case MESSAGE_REPORT:
			what = "Report";
			break;
		case MESSAGE_WARNING:
			what = "Warning";
			break;
		case MESSAGE_ASSERT:
			what = "Assert";
			stackTrace = true;
			break;
		case MESSAGE_ERROR:
			what = "Error";
			stackTrace = true;
			break;
		case MESSAGE_NONE:
		default:
			break;
	}

	showMessage(what, id, description, file, line, stackTrace);
	return msg == MESSAGE_ASSERT || msg == MESSAGE_ERROR;
}

void hkDefaultError::sectionBegin(int id, const char* sectionName)
{
	m_sectionIds.pushBack(id);
}

void hkDefaultError::sectionEnd()
{
	m_sectionIds.popBack();
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
