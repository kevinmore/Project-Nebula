/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Report/hkVdbStreamReportUtil.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>

extern "C" hkSystemTime HK_CALL hkGetSystemTime();

static void HK_CALL _writeModuleInfo(const char* text, void* context)
{
	hkOstream* stream = (hkOstream*)context;
	(*stream) << "Module( str=r'''" << text << "''' )\n";
}

// Internal structure for passing parameters to _writeStackTrace
struct _addressParams 
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE, _addressParams);

	hkOstream* m_output;
	hkArray<hkUlong>* m_addresses;
	int m_index;
};

static void HK_CALL _writeStackTrace( const char* text, void* context )
{
	_addressParams* params = (_addressParams*)context;
	hkUlong address = (*params->m_addresses)[ params->m_index ];
	int index = hkString::lastIndexOf(text, '\n');
	hkSubString subText(text, index);
	(*params->m_output) << "Location( loc=" << hkUint64(address) << ", str=r\"\"\"" << subText << "\"\"\" )\n";
	params->m_index++;
}

static void _writeMemorySystemStats(const hkTrackerScanSnapshot& scanSnapshot, hkOstream& stream)
{
	const char* prev = scanSnapshot.getMemorySystemStatistics();
	if(prev)
	{
		while( const char* cur = hkString::strChr(prev, '\n') )
		{
			stream << "Statistics( str='";
			stream.write( prev, int(cur-prev) );
			stream << "' )\n";
			cur += 1;
			prev = cur;
		}
	}
}

void HK_CALL hkVdbStreamReportUtil::generateReport( const hkTrackerScanSnapshot* scanSnapshot, hkOstream& stream )
{
	HK_ASSERT(0x543ec0e7, stream.isOk());
	// Maps to determine whether we've seen particular objects before...
	hkPointerMap<const hkTrackerTypeTreeNode*,int> seenTypes;
	hkPointerMap<hkUlong,int> seenAddresses;
	hkPointerMap<hkUlong,int> seenLocations;
	hkPointerMap<int, int> seenCallstacks;	

	hkStackTracer tracer;
	const hkMemorySnapshot& snap = scanSnapshot->getRawSnapshot();
	const hkStackTracer::CallTree& callTree = snap.getCallTree();
	const hkArray<hkTrackerScanSnapshot::Block*>& blocks = scanSnapshot->getBlocks();

	if ( blocks.getSize() && callTree.isEmpty() )
	{
		stream << "#NOTE: Could not retrieve stack information. Are you using the hkCheckingMemorySystem?\n#\n";
	}

	// Boilerplate lines describing contents of file
	
	stream << "#V <integer> - Version number\n";
	stream << "#Date(ts=<timestamp>) - Date of capture\n";
	stream << "#Module(mod=<platform-dependent module/symbol identifier string>) - Module information\n";
	stream << "#Statistics(str=<string>) - Raw memory system Statistics\n";
	stream << "#Provider(id=<provider id>, name=<name>, parIds=[<parent>*]) - Hierarchy of providers (allocators)\n";
	stream << "#Router(temp=<provider id>, stack=<provider id>, heap=<provider id>, debug=<provider id>, solver=<provider id>) - Memory router wiring\n";
	stream << "#Allocation(addr=<address>, size=<size>, provId=<provider id>, status=<status>, callstackId=<callstack id>) - Allocation report\n";
	stream << "#Type(id=<type id>, name=<type name>) - Block type definition\n";
	stream << "#Block(id=<block id>, typeId=<type id>, addr=<address>, size=<size>) - Tracker block report\n";
	stream << "#References(blockId=<block id>, refIds=[<owned block id>+]) - Blocks referenced by a given block\n";
	stream << "#Callstack(id=<callstack id>, locations=[<location>+]) - Callstack declaration for a specific address\n";
	stream << "#Location(loc=<location>, str=<string name>) - Program location\n";
	
	// v0 : tags(D,M,R,L,C,T,a,c,o)
	// v1 : adds V and A tags
	stream << "V 1\n";

	// Output timestamp
	stream << "Date( ts=" << (hkUint64)hkGetSystemTime() << " )\n";

	// Output module information required to convert addresses to lines later
	tracer.getModuleInfo( _writeModuleInfo, &stream );

	// Output raw memory system stats
	_writeMemorySystemStats(*scanSnapshot, stream);

	// Output the provider hierarchy
	{
		for( int i = 0; i < snap.m_providers.getSize(); ++i )
		{
			const hkMemorySnapshot::Provider& provider = snap.m_providers[i];
			stream.printf("Provider( id=%i, name='%s', parIds=[", i, provider.m_name );
			for( int j = 0; j < provider.m_parentIndices.getSize(); ++j )
			{
				stream.printf("%i", provider.m_parentIndices[j]);
				if(j != provider.m_parentIndices.getSize() - 1)
					stream << ',';
			}
			stream.printf("] )\n");
		}
	}

	// Output the memory router sample wiring
	stream.printf( "Router( temp=%i, stack=%i, heap=%i, debug=%i, solver=%i )\n",
		snap.m_routerWiring[0], snap.m_routerWiring[1], snap.m_routerWiring[2], snap.m_routerWiring[3], snap.m_routerWiring[4] );

	// Output the list of allocations and callstacks when the address is referred
	{
		for( int i = 0; i < snap.m_allocations.getSize(); ++i )
		{
			const hkMemorySnapshot::Allocation& a = snap.m_allocations[i];

			// Write the callstack if we haven't seen it
			if((a.m_traceId != -1) && (seenCallstacks.getWithDefault( a.m_traceId, 0 ) == 0 ))
			{					
				hkArray<hkUlong> callStack;
				int stackSize = callTree.getCallStackSize( a.m_traceId );
				callStack.setSize(stackSize);
				callTree.getCallStack(a.m_traceId, callStack.begin(), stackSize);

				// Write this callstack out
				stream << "Callstack( id=" << a.m_traceId << ", locations=[";
				for( int c = 0; c < callStack.getSize(); c++ )
				{
					stream << hkUint64(callStack[c]);
					if(c != callStack.getSize()-1)
						stream << ',';
					// Save address for output later
					seenAddresses.insert( callStack[c], 0 );
				}
				stream << "] )\n";

				seenCallstacks.insert( a.m_traceId, 1 );
			}

			stream.printf("Allocation( addr=0x%p, size=%i, provId=%i, status=%i", a.m_start, a.m_size, a.m_sourceId, int(a.m_status) );
			if(a.m_traceId != -1)
			{
				stream.printf(", callstackId=%i", a.m_traceId);
			}
			stream << " )\n";
		}
	}
	
	// Output the list of blocks with their types
	for( int i = 0; i < blocks.getSize(); i++ )
	{
		const hkTrackerScanSnapshot::Block* block = blocks[i];

		// Data potentially shared with other allocs:
		int typeId = 0;
		{
			// Write the type if we haven't seen it before
			const hkTrackerTypeTreeNode* thisType = block->m_type;
			hkResult res = seenTypes.get(thisType, &typeId);
			if( res == HK_FAILURE )
			{
				// insert new type in the map
				typeId = seenTypes.getSize();
				stream << "Type( id=" << typeId << ", name='";
				hkTrackerTypeTreeNode::dumpType(block->m_type, stream);
				stream << "' )\n";

				seenTypes.insert( thisType, typeId );
			}
		}

		// Data referring specifically to this block
		stream << "Block( id=" << i << 
			", typeId=" << typeId << 
			", addr=0x" << static_cast<const void*>(block->m_start) << 
			", size=" << static_cast<const unsigned int>(block->m_size) << " )\n";
	}	

	// Write the address-to-line table
	{
		hkArray<hkUlong> uniqueAddresses;
		uniqueAddresses.reserve( seenAddresses.getSize() );
		hkPointerMap<hkUlong,int>::Iterator iter;
		for( iter = seenAddresses.getIterator(); seenAddresses.isValid( iter ); iter = seenAddresses.getNext( iter ) )
		{
			uniqueAddresses.pushBack( seenAddresses.getKey( iter ) );
		}
		if( uniqueAddresses.getSize() )
		{
			_addressParams params;
			params.m_output = &stream;
			params.m_addresses = &uniqueAddresses;
			params.m_index = 0;
			tracer.dumpStackTrace( uniqueAddresses.begin(), uniqueAddresses.getSize(), _writeStackTrace, &params );
		}
	}
	
	// Add references data
	{
		hkPointerMap<const hkTrackerScanSnapshot::Block*,int> blockIdMap;
		for( int i = 0; i < blocks.getSize(); i++ )
		{
			blockIdMap.insert( blocks[i], i );
		}

		for( int i = 0; i < blocks.getSize(); i++ )
		{
			hkTrackerScanSnapshot::Block* cur = blocks[i];
			hkTrackerScanSnapshot::Block*const* refs = scanSnapshot->getBlockReferences( cur );
			hkArray<int> childIds;

			for( int j = 0; j < cur->m_numReferences; j++ )
			{
				int childId = blockIdMap.getWithDefault( refs[j], -1 );
				
				//HK_ASSERT2(0x543892c3, childId != -1, "Found a child block that wasn't accounted for" );
				
				if( childId != -1 )
				{	
					childIds.pushBack( childId );
				}
			}
			
			if( childIds.getSize() )
			{
				stream << "References( blockId=" << i << ", refIds=[";
				for( int c = 0; c < childIds.getSize(); c++)
				{
					stream << childIds[c];
					if(c != childIds.getSize()-1)
						stream << ',';
				}
				stream << "] )\n";
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
