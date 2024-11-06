/*----------------------------------------------------------------------------*/
/*                                                                            */
/*    Copyright (c) Innovation First 2020 All rights reserved.                */
/*    Licensed under the MIT license.                                         */
/*                                                                            */
/*    Module:     ai_jetson.cpp                                               */
/*    Author:     James Pearman                                               */
/*    Created:    3 August 2020                                               */
/*                                                                            */
/*    Revisions:  V0.1                                                        */
/*                                                                            */
/*----------------------------------------------------------------------------*/
#include "vex.h"

/*----------------------------------------------------------------------------*/
/** @file    jetson.cpp
  * @brief   class for Jetson communication
*//*--------------------------------------------------------------------------*/

using namespace vex;
using namespace ai;

/*---------------------------------------------------------------------------*/
/** @brief  Constructor                                                      */
/*---------------------------------------------------------------------------*/
// Create high priority task to handle incomming data
//
jetson::jetson() {
    state = jetson_state::kStateSyncWait1;

    thread t1 = thread( receive_task, static_cast<void *>(this) );
    t1.setPriority(thread::threadPriorityHigh);
}

jetson::~jetson() {
}

/*---------------------------------------------------------------------------*/
/** @brief  Get the total number of good received packets                    */
/*---------------------------------------------------------------------------*/
int32_t
jetson::get_packets() {
    return packets;
}
/*---------------------------------------------------------------------------*/
/** @brief  Get the total number of bad received packets                     */
/*---------------------------------------------------------------------------*/
int32_t
jetson::get_errors() {
    return errors;
}
/*---------------------------------------------------------------------------*/
/** @brief  Get the number of timeouts that have been triggered              */
/*---------------------------------------------------------------------------*/
int32_t
jetson::get_timeouts() {
    return timeouts;
}
/*---------------------------------------------------------------------------*/
/** @brief  Get the total number of bytes received                           */
/*---------------------------------------------------------------------------*/
int32_t
jetson::get_total() {
    return total_data_received;
}

/*---------------------------------------------------------------------------*/
/** @brief  Get the last received map record                                 */
/*---------------------------------------------------------------------------*/
//
// The map record is copied to user supplied buffer
// length of the valid data is returned
// 
int32_t
jetson::get_data( AI_RECORD *map ) {
    int32_t length = 0;

    if( map != NULL ) {
        maplock.lock();
        memcpy( map, &last_map, sizeof(AI_RECORD));
        length = last_payload_length;
        maplock.unlock();
    }

    return length;
}

/*----------------------------------------------------------------------------*/
/** @brief   Calculate CRC32                                                  */
/** @param  *pData        A pointer to a uint8_t array.                       */
/** @param  numberOfBytes Num of bytes on which the CRC should be calculated. */
/** @param  accumulator   zero, or the result of a previous CRC calculation.  */
/*----------------------------------------------------------------------------*/
/** @details
 *   Calculate CRC32.
 *   The accumulator should be 0 or the result of a previous CRC calculation
 */

uint32_t jetson::_crc32_table[256];

uint32_t
jetson::crc32(uint8_t *pData, uint32_t numberOfBytes, uint32_t accumulator) {
  uint32_t i, j;
  const uint32_t POLYNOMIAL_CRC32 = 0x04C11DB7;

  // On first call generate the table
  if( _crc32_table[1] == 0 ) {
    uint32_t crc_accum;

    for(i = 0; i < 256; i++) {
      crc_accum = i << 24;

      for(j = 0; j < 8; j++) {
        if(crc_accum & 0x80000000L)
          crc_accum = (crc_accum << 1) ^ POLYNOMIAL_CRC32;
        else
          crc_accum = (crc_accum << 1);
        }
      _crc32_table[i] = crc_accum;
      }  
  }

  for(j = 0; j < numberOfBytes; j++) {
    i = ((accumulator >> 24) ^ *pData++) & 0xFF;
    accumulator = (accumulator << 8) ^ _crc32_table[i];
    }

  return accumulator;
}

/*---------------------------------------------------------------------------*/
/** @brief  Parse a single received byte                                     */
/*---------------------------------------------------------------------------*/
bool
jetson::parse( uint8_t data ) {
    bool  bRecall = false;

    // 250mS interbyte timeout
    if( state != jetson_state::kStateSyncWait1 && timer.time() > 250 ) {
      timeouts++;
      state = jetson_state::kStateSyncWait1;
    }

    // reset timeout
    timer.clear();
    
    switch( state ) {
      /*----------------------------------------------------------------------*/
      // crude multi byte sync
      case jetson_state::kStateSyncWait1:
        if( static_cast<sync_byte>(data) == sync_byte::kSync1 ) {
          state = jetson_state::kStateSyncWait2;
        }
        break;

      case jetson_state::kStateSyncWait2:
        state = jetson_state::kStateSyncWait1;
        if( static_cast<sync_byte>(data) == sync_byte::kSync2 ) {
          state = jetson_state::kStateSyncWait3;
        }
        break;
        
      case jetson_state::kStateSyncWait3:
        state = jetson_state::kStateSyncWait1;
        if( static_cast<sync_byte>(data) == sync_byte::kSync3 ) {
          state = jetson_state::kStateSyncWait4;
        }
        break;
        
      case jetson_state::kStateSyncWait4:
        state = jetson_state::kStateSyncWait1;
        if( static_cast<sync_byte>(data) == sync_byte::kSync4 ) {
          state = jetson_state::kStateLength;
          index = 0;
          payload_length = 0;
        }
        break;

      /*----------------------------------------------------------------------*/
      // get payload length
      case jetson_state::kStateLength:
        // data is 2 byte little endian
        payload_length = (payload_length >> 8) + ((uint16_t)data << 8);

        if( index++ == 1 ) {
          state = jetson_state::kStateSpare;
          index = 0;
          payload_type = 0;
        }
        break;

      /*----------------------------------------------------------------------*/
      // get packet type
      case jetson_state::kStateSpare:
        // data is 2 byte little endian
        payload_type = (payload_type >> 8) + ((uint16_t)data << 8);

        if( index++ == 1 ) {
          state = jetson_state::kStateCrc32;
          index = 0;
          payload_crc32 = 0;
        }
        break;

      /*----------------------------------------------------------------------*/
      // get payload crc32
      case jetson_state::kStateCrc32:
        // data is 4 byte little endian
        payload_crc32 = (payload_crc32 >> 8) + ((uint32_t)data << 24);
        
        if( index++ == 3 ) {
          state = jetson_state::kStatePayload;
          index = 0;
          calc_crc32 = 0;
        }
        break;

      /*----------------------------------------------------------------------*/
      // get payload data
      case jetson_state::kStatePayload:
        if( index < sizeof(payload) ) {
          // add byte to buffer
          payload.bytes[index] = data;
          index++;

          // keep running crc32, save calculating all at once later
          calc_crc32 = crc32( &data, 1, calc_crc32  );
          
          // all data received ?
          if( index == payload_length ) {
            // check crc32
            if( payload_crc32 == calc_crc32 ) {
              state = jetson_state::kStateGoodPacket;
              bRecall = true;
            }
            else {
              state = jetson_state::kStateBadPacket;
              bRecall = true;
            }
          }
        }
        else {
          // if we end up here then error
          //
          state = jetson_state::kStateBadPacket;
          bRecall = true;
        }
        break;

      case jetson_state::kStateGoodPacket:
        if( payload_type == MAP_PACKET_TYPE ) {
          AI_RECORD newMap;
          // Parse the payload packet into a AI_RECORD
          memset(&newMap, 0, sizeof(newMap));
          memcpy(&newMap, &payload.bytes[0], MAP_POS_SIZE);
          if(newMap.detectionCount > MAX_DETECTIONS)
            newMap.detectionCount = MAX_DETECTIONS;
          memcpy(&newMap.detections, &payload.bytes[MAP_POS_SIZE], sizeof(DETECTION_OBJECT) * newMap.detectionCount);


          // lock access to last_map and copy data
          maplock.lock();
          memcpy( &last_map, &newMap, sizeof(AI_RECORD));
          maplock.unlock();
        }

        // timestamp this packet
        last_packet_time = timer.system();

        packets++;
        state = jetson_state::kStateSyncWait1;
        break;

      case jetson_state::kStateBadPacket:
        // bad packet
        errors++;
        state = jetson_state::kStateSyncWait1;
        break;

      default:
        state = jetson_state::kStateSyncWait1;
        break;
    }

    return bRecall;
}

/*---------------------------------------------------------------------------*/
/** @brief  Send request to the Jetson to ask for next packet                */
/*---------------------------------------------------------------------------*/
void
jetson::request_map() {
    // check timeout and clear state machine if necessary
    if( state != jetson_state::kStateSyncWait1 && timer.time() > 250 ) {
      state = jetson_state::kStateSyncWait1;
    }

    // only send if we are waiting for a message
    if( state == jetson_state::kStateSyncWait1 ) {
      // Send small message to jetson asking for data
      // using simple C stdio for simplicity
      // serial1 is the second USB cdc channel dedicated to user programs
      // we use this rather than stdout so linefeed is not expanded
      //
      FILE *fp = fopen("/dev/serial1", "w");

      // This is arbitary message at the moment
      // just using ASCII for convienience and debug porposes
      //
      static char msg[] = "AA55CC3301\r\n"; 

      // send
      fwrite( msg, 1, strlen(msg), fp );

      // close, flush buffers
      fclose(fp);
    }
}

/*---------------------------------------------------------------------------*/
/** @brief  Task to receive and process receive data from Jetson             */
/*---------------------------------------------------------------------------*/
int
jetson::receive_task( void *arg ) {
    if( arg == NULL)
      return(0);
      
    // get our Jetsn instance
    // theoretically we could handle more than one channel on different
    // ports, but that's beyond the scope of this demo code
    //
    jetson *instance = static_cast<jetson *>(arg);

    // process one character at a time
    // getchar() is blocking and will call yield internally
    //
    while(1) {
      // this will block
      int rxchar = getchar();

      // process this byte
      if( rxchar >= 0 ) {
        instance->total_data_received++;
    
        // parse returns true if there is more processing to do
        while( instance->parse( (uint8_t)rxchar ) )
          this_thread::yield();
      }

      // no need for sleep/yield as stdin is blocking
    }
}
