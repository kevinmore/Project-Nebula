/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/BlockStream/hkBlockStream.h>
#include <Common/Base/Container/CommandStream/hkCommandStream.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>


hkBlockStreamCommandWriter::hkBlockStreamCommandWriter( )
{
}


#if !defined(HK_PLATFORM_SPU)
void hkBlockStreamCommandWriter::append( const hkCommand& command )
{
	int numBytes = command.m_sizePaddedTo16;
	hkCommand *dst = m_writer.reserve( numBytes );
#if 1 || defined(HK_PLATFORM_WIN32)
	hkString::memCpy4(dst, &command, numBytes>>2);
#else
	hkString::memCpy16NonEmpty(dst, &command, numBytes>>4);	// commands will not be aligned on the stack
#endif
	m_writer.advance( numBytes );
}
#endif

void hkBlockStreamCommandWriter::exec( const hkCommand& command )
{
	int numBytes = command.m_sizePaddedTo16;
	hkCommand *dst = m_writer.reserve( numBytes );
	hkString::memCpy4(dst, &command, numBytes>>2);
	m_writer.advance( numBytes );
}


#if !defined(HK_PLATFORM_SPU)
class ErrorCommandDispatcher: public hkSecondaryCommandDispatcher
{
public:
	virtual void exec( const hkCommand& command )
	{
		HK_ASSERT3( 0xf0345456, false, "Unhandled command of type " << command.m_primaryType );
	}
	virtual void print( const hkCommand& command, hkOstream& stream ) const
	{
		HK_ASSERT3( 0xf0345456, false, "Unhandled command of type " << command.m_primaryType );
	}
} g_errorDispatcher;

hkPrimaryCommandDispatcher::hkPrimaryCommandDispatcher()
{
	for (int i=0; i < hkCommand::TYPE_MAX; i++ )
	{
		m_commandDispatcher[i] = &g_errorDispatcher; 
	}
}

hkPrimaryCommandDispatcher::~hkPrimaryCommandDispatcher()
{
	for (int i=0; i < hkCommand::TYPE_MAX; i++ )
	{
		//m_commandDispatcher[i]->removeReference();
	}
}

void hkPrimaryCommandDispatcher::registerDispatcher( hkCommand::PrimaryType type, hkSecondaryCommandDispatcher* dispatcher )
{
	m_commandDispatcher[type] = dispatcher;
}


void hkPrimaryCommandDispatcher::exec( const hkCommand& command )
{
	HK_ASSERT2( 0xf0de341d, command.m_primaryType < hkCommand::TYPE_MAX, "Unknown command type" );
	m_commandDispatcher[ command.m_primaryType ]->exec( command );
}

void hkPrimaryCommandDispatcher::print( const hkCommand& command, hkOstream& stream )
{
	HK_ASSERT2( 0xf0de341d, command.m_primaryType < hkCommand::TYPE_MAX, "Unknown command type" );
	m_commandDispatcher[ command.m_primaryType ]->print( command, stream );
}

#endif // !SPU

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
