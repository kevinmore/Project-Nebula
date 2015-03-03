/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Serialize/Util/hkStructureLayout.h>

#if defined(HK_COMPILER_MSVC) || defined(HK_COMPILER_INTEL)
#	define HK_ALIGN_OF(T) __alignof(T)
#elif defined(HK_COMPILER_GCC) || defined(HK_COMPILER_SNC) || defined(HK_COMPILER_CLANG)
#	define HK_ALIGN_OF(T) __alignof__(T)
#elif defined(HK_COMPILER_MWERKS)
#	define HK_ALIGN_OF(T) __builtin_align(T)
#else
#	error fixme
#endif

// detect the host properties at compile time.

namespace hkSerializerHostDetection
{
	struct EmptyBase
	{
	};

	struct InheritsEmptyBase : public EmptyBase
	{
		int x;
	};

	struct EndPadding
	{

		struct Quad
		{
			Quad() { }
			HK_ALIGN16( int x[4] );
		};

		Quad q;
		int y;
	};

	struct InheritsEndPadding : public EndPadding
	{
		int z;
	};

	static const union { char c[4]; int i; } endianChecker = { { 1, 0, 0, 0 } };

	struct VirtualClass
	{
		virtual ~VirtualClass();
		virtual void doit();
		int data;
	};

		// some compilers (notably gcc2.x put the vtable after data members.
	static const int vtableAlwaysFirst = HK_OFFSET_OF(VirtualClass, data) == sizeof(void*);
}

const hkStructureLayout::LayoutRules hkStructureLayout::HostLayoutRules =
{
	sizeof(void*), // bytes in pointer
	hkSerializerHostDetection::endianChecker.i == 1, // littleEndian
	HK_OFFSET_OF( hkSerializerHostDetection::InheritsEndPadding, z ) != sizeof(hkSerializerHostDetection::EndPadding), // reusePaddingOptimization;
	HK_OFFSET_OF( hkSerializerHostDetection::InheritsEmptyBase, x ) == 0 // emptyBaseClassOptimization
};

hkStructureLayout::hkStructureLayout()
:	m_rules(HostLayoutRules)
{
}

// msvc
const hkStructureLayout::LayoutRules hkStructureLayout::MsvcWin32LayoutRules = {4,1,0,1};
const hkStructureLayout::LayoutRules hkStructureLayout::MsvcXboxLayoutRules = {4,1,0,1};
const hkStructureLayout::LayoutRules hkStructureLayout::MsvcAmd64LayoutRules = {8,1,0,1};
// codewarrior
const hkStructureLayout::LayoutRules hkStructureLayout::CwPs2LayoutRules = {4,1,0,1};
const hkStructureLayout::LayoutRules hkStructureLayout::CwNgcLayoutRules = {4,0,0,1};
const hkStructureLayout::LayoutRules hkStructureLayout::CwPspLayoutRules = {4,1,0,1};
const hkStructureLayout::LayoutRules hkStructureLayout::CwWiiLayoutRules = {4,0,1,1};
// gcc
const hkStructureLayout::LayoutRules hkStructureLayout::Gcc32Ps2LayoutRules = {4,1,1,1};
const hkStructureLayout::LayoutRules hkStructureLayout::Gcc151PspLayoutRules = {4,1,1,1};
const hkStructureLayout::LayoutRules hkStructureLayout::Gcc295LinuxLayoutRules = {4,1,0,0};
const hkStructureLayout::LayoutRules hkStructureLayout::Gcc33LinuxLayoutRules = {4,1,1,1};
const hkStructureLayout::LayoutRules hkStructureLayout::GccAmd64LinuxLayoutRules = {8,1,1,1};
const hkStructureLayout::LayoutRules hkStructureLayout::GccAndroidLayoutRules = {4,1,1,1};
// sn
const hkStructureLayout::LayoutRules hkStructureLayout::Sn31Ps2LayoutRules = {4,1,0,0};
const hkStructureLayout::LayoutRules hkStructureLayout::Sn10PspLayoutRules = {4,1,0,1};
const hkStructureLayout::LayoutRules hkStructureLayout::Sn393NgcLayoutRules = {4,0,0,0};
const hkStructureLayout::LayoutRules hkStructureLayout::SnPsVitaLayoutRules = {4,1,1,1};
// x360
const hkStructureLayout::LayoutRules hkStructureLayout::Xbox360LayoutRules = {4,0,0,1};
// PlayStation(R)3
const hkStructureLayout::LayoutRules hkStructureLayout::GccPs3LayoutRules = {4,0,1,1};
// mac
const hkStructureLayout::LayoutRules hkStructureLayout::Gcc40MacPpcLayoutRules = {4,0,1,1};
const hkStructureLayout::LayoutRules hkStructureLayout::Gcc40MacIntelLayoutRules = {4,1,1,1};
const hkStructureLayout::LayoutRules hkStructureLayout::Gcc40iOSLayoutRules = {4,1,1,1};
// 3DS
const hkStructureLayout::LayoutRules hkStructureLayout::Rvct3dsLayoutRules = {4,1,1,1};
// Wii U
const hkStructureLayout::LayoutRules hkStructureLayout::GhsWiiULayoutRules = {4,0,0,1};
// PlayStation(R)4
const hkStructureLayout::LayoutRules hkStructureLayout::Ps4LayoutRules = {8,1,1,1};
// XBox One
const hkStructureLayout::LayoutRules hkStructureLayout::XboxOneLayoutRules = {8,1,0,1};

hkStructureLayout::hkStructureLayout( const LayoutRules& rules )
	:	m_rules(rules)
{
}

static int computeMemberSize(
	const hkClassMember& member,
	const hkStructureLayout::LayoutRules& rules,
	const hkClassMember::Type type )
{
	int size = 0;
	switch( type )
	{
		case hkClassMember::TYPE_BOOL:
		case hkClassMember::TYPE_CHAR:
		case hkClassMember::TYPE_INT8:
		case hkClassMember::TYPE_UINT8:
		case hkClassMember::TYPE_INT16:
		case hkClassMember::TYPE_UINT16:
		case hkClassMember::TYPE_INT32:
		case hkClassMember::TYPE_UINT32:
		case hkClassMember::TYPE_INT64:
		case hkClassMember::TYPE_UINT64:
		case hkClassMember::TYPE_REAL:
		case hkClassMember::TYPE_HALF:
		case hkClassMember::TYPE_VECTOR4:
		case hkClassMember::TYPE_QUATERNION:
		case hkClassMember::TYPE_MATRIX3:
		case hkClassMember::TYPE_ROTATION:
		case hkClassMember::TYPE_QSTRANSFORM:
		case hkClassMember::TYPE_MATRIX4:
		case hkClassMember::TYPE_TRANSFORM:
		case hkClassMember::TYPE_ENUM:
		case hkClassMember::TYPE_FLAGS:
		{
			size = member.getSizeInBytes();
			break;
		}
		case hkClassMember::TYPE_ULONG:
		case hkClassMember::TYPE_CSTRING:
		case hkClassMember::TYPE_STRINGPTR:
		case hkClassMember::TYPE_POINTER:
		case hkClassMember::TYPE_FUNCTIONPOINTER:
		{
			size = rules.m_bytesInPointer;
			size *= (member.getCstyleArraySize()) ? member.getCstyleArraySize() : 1;
			break;
		}
		case hkClassMember::TYPE_VARIANT:
		{
			size = rules.m_bytesInPointer * 2;
			break;
		}
		case hkClassMember::TYPE_RELARRAY:
		{
			size = 2 * hkSizeOf(hkUint16);
			break;
		}
		case hkClassMember::TYPE_ARRAY:
		{
			size = sizeof(hkUint32);
			// fallthrough
		}
		case hkClassMember::TYPE_SIMPLEARRAY:
		{
			size += rules.m_bytesInPointer + sizeof(hkUint32);
			break;
		}
		case hkClassMember::TYPE_STRUCT:
		{
			const hkClass& sclass = member.getStructClass();
			size = sclass.getObjectSize();
			size *= (member.getCstyleArraySize()) ? member.getCstyleArraySize() : 1;
			break;
		}
		case hkClassMember::TYPE_ZERO:
		{
			size = computeMemberSize( member, rules, member.getSubType() );
			break;
		}
		case hkClassMember::TYPE_HOMOGENEOUSARRAY:
		{
			// class pointer + data pointer + size
			size = 2 * rules.m_bytesInPointer + sizeof(hkUint32);
			break;
		}
		case hkClassMember::TYPE_INPLACEARRAY:
		{
			HK_ASSERT2(0x50a18b57, 0, "Inplace Array not currently handled in structure layout computations.");
			break;
		}
		default:
		{
			HK_ERROR(0x50a18b58, "Unknown class member type in structureLayout::getMemberSize().");
		}
	}

	return size;
}

static int getLayoutAlignment(const hkClassMember& member, hkClassMember::Type memType, int bytesInPointer )
{
	int align = -1;
	switch( memType )
	{
		case hkClassMember::TYPE_VOID:
		case hkClassMember::TYPE_BOOL:
		case hkClassMember::TYPE_CHAR:
		case hkClassMember::TYPE_INT8:
		case hkClassMember::TYPE_UINT8:
		case hkClassMember::TYPE_INT16:
		case hkClassMember::TYPE_UINT16:
		case hkClassMember::TYPE_INT32:
		case hkClassMember::TYPE_UINT32:
		case hkClassMember::TYPE_INT64:
		case hkClassMember::TYPE_UINT64:
		case hkClassMember::TYPE_REAL:
		case hkClassMember::TYPE_HALF:
		case hkClassMember::TYPE_VECTOR4:
		case hkClassMember::TYPE_QUATERNION:
		case hkClassMember::TYPE_MATRIX3:
		case hkClassMember::TYPE_ROTATION:
		case hkClassMember::TYPE_QSTRANSFORM:
		case hkClassMember::TYPE_MATRIX4:
		case hkClassMember::TYPE_TRANSFORM:
		case hkClassMember::TYPE_ENUM:
		case hkClassMember::TYPE_FLAGS:
		case hkClassMember::TYPE_RELARRAY:
		{
			align = member.getAlignment(); // same on all platforms
			break;
		}
		case hkClassMember::TYPE_ULONG:
		case hkClassMember::TYPE_CSTRING:
		case hkClassMember::TYPE_STRINGPTR:
		case hkClassMember::TYPE_POINTER:
		case hkClassMember::TYPE_FUNCTIONPOINTER:
		case hkClassMember::TYPE_ARRAY:
		case hkClassMember::TYPE_SIMPLEARRAY:
		case hkClassMember::TYPE_HOMOGENEOUSARRAY:
		case hkClassMember::TYPE_VARIANT:
		{
			align = bytesInPointer;
			break;
		}
		case hkClassMember::TYPE_ZERO:
		{
			align = getLayoutAlignment( member, member.getSubType(), bytesInPointer);
			break;
		}
		case hkClassMember::TYPE_STRUCT:
		{
			HK_ASSERT(0x2a66fb73, member.hasClass() );
			int biggestAlign = 1;
			const hkClass& sclass = member.getStructClass();
			for( int i = 0; i < sclass.getNumMembers(); ++i )
			{
				const hkClassMember& m = sclass.getMember(i);
				int a = getLayoutAlignment(m, m.getType(), bytesInPointer);
				if( a > biggestAlign )
				{
					biggestAlign = a;
				}
			}
			align = biggestAlign;
			break;
		}
		case hkClassMember::TYPE_INPLACEARRAY: //XXX depends on template
		case hkClassMember::TYPE_MAX:
		{
			HK_ASSERT(0x4a8dd63d,0);
			break;
		}
	}
	HK_ASSERT(0x6b0ba3d6, align != -1);
	if( int extra = member.getFlags().get( hkClassMember::ALIGN_32|hkClassMember::ALIGN_16|hkClassMember::ALIGN_8 ) )
	{
		align = hkMath::max2( 
			(extra == hkClassMember::ALIGN_32) ? 32 : (extra == hkClassMember::ALIGN_16) ? 16 : 8, 
			align 
		);
	}

	return align;
}

static HK_NEVER_INLINE void s_setOffset( hkClassMember& member, int currentOffset )
{
	member.setOffset( currentOffset );
}

static void retargetClassInplace(
	hkClass& topLevelClass,
	const hkStructureLayout::LayoutRules& rules,
	hkPointerMap<const hkClass*,int>& classesDone,
	hkBool32 recurse )
{
	HK_ASSERT( 0x8301838b, classesDone.hasKey( &topLevelClass ) == false );
	classesDone.insert( &topLevelClass, 0 );

	// Chase used member classes depth first
	if( recurse )
	{
		int numMembers = topLevelClass.getNumMembers();
		int memberIndex;
		for( memberIndex = 0; memberIndex < numMembers; ++memberIndex )
		{
			hkClassMember& mem = topLevelClass.getMember(memberIndex);
			if( mem.hasClass() && classesDone.hasKey(&mem.getStructClass()) == false )
			{
				retargetClassInplace( const_cast<hkClass&>(mem.getStructClass()), rules, classesDone, recurse );
			}
		}
	}

	// Traverse from base to most derived
	hkArray<hkClass*>::Temp hier; hier.reserve(16);
	{
		const hkClass* cl = &topLevelClass;
		while( cl )
		{
			hier.insertAt(0, const_cast<hkClass*>(cl) );
			cl = cl->getParent();
		}
	}

	// This vtable may not be where we expect it
	HK_ASSERT3(0x55e6397d, topLevelClass.hasVtable() == false
		|| hkSerializerHostDetection::vtableAlwaysFirst || hier[0]->getNumMembers() == 0,
		"The vtable for "<<topLevelClass.getName()<<" is not at the start of the class." );
	HK_ASSERT2(0x45797d50, topLevelClass.hasVtable() == hier[0]->hasVtable(),
		"Virtual class inheriting from nonvirtual class not supported");

	int currentOffset = 0;
	int biggestAlignSoFar = 1;

	const int alignmentOfPointer = rules.m_bytesInPointer;
	const int sizeOfPointer = rules.m_bytesInPointer;
	HK_COMPILE_TIME_ASSERT( HK_ALIGN_OF(void*) == sizeof(void*) );

	for( int hierIdx = 0; hierIdx < hier.getSize(); ++hierIdx)
	{
		hkClass& klass = *hier[hierIdx];
		
		for( int interfaceIdx = 0; interfaceIdx < klass.getNumDeclaredInterfaces(); ++interfaceIdx )
		{
			if( currentOffset % alignmentOfPointer != 0 )
			{
				currentOffset -= currentOffset % alignmentOfPointer;
				currentOffset += alignmentOfPointer;
			}
			currentOffset += sizeOfPointer;
			if( alignmentOfPointer > biggestAlignSoFar )
			{
				biggestAlignSoFar = alignmentOfPointer;
			}
		}

		for( int memIdx = 0; memIdx < klass.getNumDeclaredMembers(); ++memIdx)
		{
			hkClassMember& member = const_cast<hkClassMember&>(klass.getDeclaredMember(memIdx));

			// maybe make room for empty base class
			if( (currentOffset==0)
				&& (hierIdx != 0)
				&& (rules.m_emptyBaseClassOptimization == 0) )
			{
				currentOffset = 1; // automatically adjusted by alignment below
			}

			// maybe adjust alignment
			int memberAlign = getLayoutAlignment(member, member.getType(), rules.m_bytesInPointer);

			// save biggest alignment for possible trailing padding
			if( memberAlign > biggestAlignSoFar )
			{
				biggestAlignSoFar = memberAlign;
			}

			// padding before member?
			if( currentOffset % memberAlign != 0 )
			{
				currentOffset -= currentOffset % memberAlign;
				currentOffset += memberAlign;
			}

			// save the offset
			if( member.getOffset() != currentOffset )
			{
				s_setOffset( member, currentOffset );
			}

			// adjust for next member
			currentOffset += computeMemberSize( member, rules, member.getType() );
		}
		// padding at end of class?
		int paddedOffset = currentOffset;
		if (paddedOffset % biggestAlignSoFar != 0 )
		{
			paddedOffset -= paddedOffset % biggestAlignSoFar;
			paddedOffset += biggestAlignSoFar;
		}
		// sizes out are the padded sizes
		// sizeof(empty class) == 1
		int objectSize = paddedOffset ? paddedOffset : 1;
		if( klass.getObjectSize() != objectSize )
		{
			klass.setObjectSize(objectSize);
		}

		if( rules.m_reusePaddingOptimization == 0)
		{
			currentOffset = paddedOffset;
		}
	}
}

void hkStructureLayout::computeMemberOffsetsInplace( hkClass& klass, hkPointerMap<const hkClass*,int>& classesDone, hkBool32 recurse ) const
{
	if( classesDone.hasKey( &klass ) == false )
	{
		retargetClassInplace( klass, m_rules, classesDone, recurse );
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
