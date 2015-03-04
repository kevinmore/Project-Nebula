/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/DebugUtil/Logger/hkLogger.h>
#include <Common/Base/Fwd/hkcstdarg.h>

using namespace std;
namespace
{
	class NullLogger : public hkLogger
	{
		public:

			virtual void pushScope(const char* name) {}
			virtual void popScope() {}
			virtual void setThreshold( int level ) {}
			virtual void _log( int level, const char* fmt, hk_va_list args) {}
			virtual void flush() { }
	};
}

hkLogger& hkLogger::nullLogger()
{
	static NullLogger instance;
	return instance;
}

hkLogger::~hkLogger()
{
}

#define _LOG(LEVEL, FMT) \
	va_list args; \
	va_start(args, FMT); \
	_log( LEVEL, FMT, args ); \
	va_end(args);

void hkLogger::error( const char* fmt, ... )
{
	_LOG(LOG_ERROR, fmt);
}
void hkLogger::warning( const char* fmt, ... )
{
	_LOG(LOG_WARNING, fmt);
}
void hkLogger::debug( const char* fmt, ... )
{
	_LOG(LOG_DEBUG, fmt);
}
void hkLogger::info( const char* fmt, ... )
{
	_LOG(LOG_INFO, fmt);
}
void hkLogger::log( int level, const char* fmt, ... )
{
	_LOG(level, fmt);
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
