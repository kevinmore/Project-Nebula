/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Config/hkOptionalComponent.h>

static hkOptionalComponent* s_components;

hkOptionalComponent::hkOptionalComponent(const char* name, hkOptionalComponent::OnRequestFunction onLink)
	: m_next(s_components)
	, m_name(name)
	, m_onLink(onLink)
	, m_funcPtr(HK_NULL)
	, m_func(HK_NULL)
	, m_isUsed(false)
	, m_isRequested(false)
{
	s_components = this;
}

hkOptionalComponent::hkOptionalComponent(const char* name, void** funcPtr, void* func)
	: m_next(s_components)
	, m_name(name)
	, m_onLink(HK_NULL)
	, m_funcPtr(funcPtr)
	, m_func(func)
	, m_isUsed(false)
	, m_isRequested(false)
{
	s_components = this;
}

const hkOptionalComponent* hkOptionalComponent::getFirstComponent()
{
	return s_components;
}

void hkOptionalComponent::request()
{
	m_isRequested = true;
	if( m_onLink )
	{
		(*m_onLink)();
	}
	if( m_funcPtr )
	{
		if( *m_funcPtr != HK_NULL && *m_funcPtr != m_func )
		{
			HK_WARN(0x4f6156c6, "Optional component '" << m_name << "' has been overwritten.\n" \
				"Perhaps you have registered more than one component initialize the same creation function?");
		}
		*m_funcPtr = m_func;
	}
}

void hkOptionalComponent::writeReport(hkOstream& os)
{
	for( int i = 0; i < 4; ++i ) // i is a 2 bit bitfield of states
	{
		typedef const char* string;
		static string header[4] = // be careful with the commas between strings!
		{
			// -used -request
			"Linked but not requested\n"
			"The linker added these classes to the executable even though they were not requested by the hkOptionalComponents\n"
			"There is probably a hard dependency somewhere"
			,
			// -used +request
			"Requested but not used\n"
			"It may be that the code path which uses this component was not triggered or\n"
			"you may be able to reduce code size by not requesting them."
			,
			// +used -request
			"Not requested but used\n"
			"These were not requested but somehow marked as used. This can happen by calling the creation function directly instead of using the function pointer."
			,
			// +used +request
			"Requested and Used\n"
			"This is the normal case",
		};
		bool printedHeader = false;
		for( const hkOptionalComponent* cur = hkOptionalComponent::getFirstComponent(); cur != HK_NULL; cur = cur->getNext() )
		{
			int state = (int(cur->isUsed()<<1)) | int(cur->wasRequested());
			if( state == i )
			{
				if( printedHeader == false )
				{
					printedHeader = true;
					os.printf("\n\n========================================================\n");
					os.printf("%s\n", header[i]);
					os.printf("--------------------------------------------------------\n");
				}
				os.printf("%s\n", cur->getName());
			}
		}
	}
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
